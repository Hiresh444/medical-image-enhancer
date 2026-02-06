interface Props {
  data: unknown
  title?: string
}

export default function JsonViewer({ data, title }: Props) {
  let parsed = data
  if (typeof data === 'string') {
    try {
      parsed = JSON.parse(data)
    } catch {
      // keep as-is
    }
  }

  const text = typeof parsed === 'string' ? parsed : JSON.stringify(parsed, null, 2)

  return (
    <div>
      {title && <h3 className="mb-1">{title}</h3>}
      {!text || text === '""' || text === '{}' || text === '' ? (
        <p className="text-muted">No data available (deterministic mode — no GenAI plan).</p>
      ) : (
        <pre>{text}</pre>
      )}
    </div>
  )
}
