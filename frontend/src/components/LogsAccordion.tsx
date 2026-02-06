import { useState } from 'react'
import type { LogEntry } from '../api/client'

interface Props {
  logs: LogEntry[]
}

export default function LogsAccordion({ logs }: Props) {
  const [openIdx, setOpenIdx] = useState<number | null>(null)

  if (!logs || logs.length === 0) {
    return <p className="text-muted">No agent logs recorded for this run.</p>
  }

  return (
    <div>
      {logs.map((entry, i) => {
        const isOpen = openIdx === i
        const header = `${entry.phase ?? 'unknown'} — ${entry.event ?? ''}`
        return (
          <div className="accordion-item" key={i}>
            <div
              className="accordion-header"
              onClick={() => setOpenIdx(isOpen ? null : i)}
            >
              <span>
                <span className="badge badge-info" style={{ marginRight: '0.5rem' }}>
                  {entry.event ?? '?'}
                </span>
                {entry.phase ?? 'trace'}
                {entry.timestamp && (
                  <span className="text-muted" style={{ marginLeft: '0.75rem', fontSize: '0.8rem' }}>
                    {entry.timestamp}
                  </span>
                )}
              </span>
              <span>{isOpen ? '▾' : '▸'}</span>
            </div>
            {isOpen && (
              <div className="accordion-body">
                <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                  {typeof entry.detail === 'string'
                    ? entry.detail
                    : JSON.stringify(entry.detail, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
