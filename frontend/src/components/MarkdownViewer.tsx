import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

interface Props {
  markdown: string
}

export default function MarkdownViewer({ markdown }: Props) {
  if (!markdown) {
    return <p className="text-muted">No report content available.</p>
  }
  return (
    <div className="markdown-body">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{markdown}</ReactMarkdown>
    </div>
  )
}
