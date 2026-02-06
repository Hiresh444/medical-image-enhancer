import { useState, useRef, useEffect, useCallback } from 'react'
import { sendChat, type ChatMessage } from '../api/client'

interface Props {
  runId: string
  initialHistory: ChatMessage[]
}

export default function ChatPanel({ runId, initialHistory }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>(initialHistory)
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const msgsEnd = useRef<HTMLDivElement>(null)

  useEffect(() => {
    msgsEnd.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = useCallback(async () => {
    const msg = input.trim()
    if (!msg || sending) return
    setInput('')
    setMessages((prev) => [...prev, { role: 'user', content: msg }])
    setSending(true)
    try {
      const reply = await sendChat(runId, msg)
      setMessages((prev) => [...prev, { role: 'assistant', content: reply }])
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `Error: ${err instanceof Error ? err.message : 'unknown'}` },
      ])
    } finally {
      setSending(false)
    }
  }, [runId, input, sending])

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.length === 0 && (
          <p className="text-muted" style={{ padding: '2rem', textAlign: 'center' }}>
            Ask a question about this run...
          </p>
        )}
        {messages.map((m, i) => (
          <div className="chat-msg" key={i}>
            <div className={`role ${m.role}`}>
              {m.role === 'user' ? 'You' : 'Assistant'}
            </div>
            <div className="content">{m.content}</div>
          </div>
        ))}
        {sending && (
          <div className="chat-msg">
            <div className="role assistant">Assistant</div>
            <div className="content text-muted">Typing...</div>
          </div>
        )}
        <div ref={msgsEnd} />
      </div>
      <div className="chat-input-row">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && send()}
          placeholder="Ask about this run..."
          disabled={sending}
          maxLength={2000}
        />
        <button className="btn btn-primary" onClick={send} disabled={sending || !input.trim()}>
          Send
        </button>
      </div>
    </div>
  )
}
