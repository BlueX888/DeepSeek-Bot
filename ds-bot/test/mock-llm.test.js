import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { mkdtempSync, rmSync } from 'node:fs'
import { createServer } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { after, test } from 'node:test'

// dev/mock-llm.mjs is a script that starts listening as soon as it is imported and
// exports nothing, so its commands are tested over HTTP against a spawned copy,
// as the test bench runs it. One server for the whole file, on a free port.
const freePort = () => new Promise((resolve, reject) => {
  const probe = createServer()
  probe.listen(0, '127.0.0.1', () => {
    const port = probe.address().port
    probe.close(() => resolve(port))
  })
  probe.on('error', reject)
})
const PORT = await freePort()
const logDir = mkdtempSync(join(tmpdir(), 'ds-bot-mock-llm-'))
const mock = spawn(process.execPath, ['dev/mock-llm.mjs'], {
  cwd: fileURLToPath(new URL('..', import.meta.url)),
  env: { ...process.env, MOCK_PORT: String(PORT), MOCK_DELAY_MS: '1', MOCK_LOG: join(logDir, 'mock.jsonl') },
  stdio: ['ignore', 'pipe', 'pipe'],
})
after(() => { mock.kill(); rmSync(logDir, { recursive: true, force: true }) })
await new Promise((resolve, reject) => {
  const timer = setTimeout(() => reject(new Error('the mock model did not start')), 10_000)
  mock.stdout.once('data', () => { clearTimeout(timer); resolve() })
  mock.once('exit', code => reject(new Error(`the mock model exited with ${code}`)))
  mock.once('error', reject)
})

// One streamed turn, in the Anthropic event framing the plugin sends and reads.
async function turn(messages) {
  const response = await fetch(`http://127.0.0.1:${PORT}/v1/messages`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: 'mock', stream: true, messages }),
    signal: AbortSignal.timeout(20_000),
  })
  assert.equal(response.status, 200)
  const events = [...(await response.text()).matchAll(/^data: (.+)$/gm)].map(match => JSON.parse(match[1]))
  const blocks = []
  for (const event of events) {
    if (event.type === 'content_block_start') blocks.push({ ...event.content_block, index: event.index })
    else if (event.type === 'content_block_delta' && event.delta.type === 'text_delta') blocks[event.index].text = (blocks[event.index].text ?? '') + event.delta.text
    else if (event.type === 'content_block_delta' && event.delta.type === 'input_json_delta') blocks[event.index].input = JSON.parse(event.delta.partial_json)
    else if (event.type === 'message_delta') blocks.stop_reason = event.delta.stop_reason
  }
  return blocks
}

test('a 长任务 turn grows every step even with K left at its default', async () => {
  // The opener starts a two-step turn and answers the first block of filler itself.
  const opener = await turn([{ role: 'user', content: '长任务 2' }])
  const step0 = opener.find(block => block.type === 'text')
  assert.equal(step0.text.length, 4000)
  assert.match(step0.text, /^步骤0第1段/)
  const call0 = opener.find(block => block.type === 'tool_use')
  assert.equal(call0.name, 'list_bots')

  // The first tool result comes back: the next step must add its own filler (default
  // K = 4), not answer with no text at all.
  const step = await turn([
    { role: 'user', content: '长任务 2' },
    { role: 'assistant', content: [step0, call0] },
    { role: 'user', content: [{ type: 'tool_result', tool_use_id: call0.id, content: '[]' }] },
  ])
  const grown = step.find(block => block.type === 'text')
  assert.ok(grown, 'the step answers a text block')
  assert.equal(grown.text.length, 4000)
  assert.match(grown.text, /^步骤2第1段/)
  const call1 = step.find(block => block.type === 'tool_use')
  assert.equal(call1.name, 'list_bots')

  // After the second tool result the turn reports that it is done.
  const done = await turn([
    { role: 'user', content: '长任务 2' },
    { role: 'assistant', content: [step0, call0] },
    { role: 'user', content: [{ type: 'tool_result', tool_use_id: call0.id, content: '[]' }] },
    { role: 'assistant', content: [grown, call1] },
    { role: 'user', content: [{ type: 'tool_result', tool_use_id: call1.id, content: '[]' }] },
  ])
  assert.equal(done.find(block => block.type === 'text')?.text, '长任务做完了，一共 2 步。')
  assert.equal(done.stop_reason, 'end_turn')
})
