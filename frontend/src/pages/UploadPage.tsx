import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import FileUpload from '../components/FileUpload'
import { uploadFile, startRun, pollStatus } from '../api/client'

const MODELS = [
  { value: 'gpt-5-mini', label: 'GPT-5 Mini (default)' },
  { value: 'gpt-4.1-mini', label: 'GPT-4.1 Mini' },
  { value: 'gpt-4.1', label: 'GPT-4.1' },
  { value: 'o4-mini', label: 'o4-mini' },
]

type Phase = 'idle' | 'uploading' | 'starting' | 'polling' | 'done' | 'error'

export default function UploadPage() {
  const navigate = useNavigate()
  const [file, setFile] = useState<File | null>(null)
  const [genai, setGenai] = useState(false)
  const [model, setModel] = useState('gpt-5-mini')
  const [maxIters, setMaxIters] = useState(2)
  const [phase, setPhase] = useState<Phase>('idle')
  const [status, setStatus] = useState('')
  const [error, setError] = useState('')
  const [runId, setRunId] = useState('')

  const handleRun = useCallback(async () => {
    if (!file) return
    setError('')
    try {
      // 1. Upload
      setPhase('uploading')
      setStatus('Uploading file...')
      const upload = await uploadFile(file)

      // 2. Start run
      setPhase('starting')
      setStatus('Starting pipeline...')
      const run = await startRun({
        file_id: upload.file_id,
        genai,
        model: genai ? model : undefined,
        max_iters: genai ? maxIters : undefined,
      })
      setRunId(run.run_id)

      // 3. Poll
      setPhase('polling')
      setStatus(`Run ${run.run_id} — waiting for completion...`)

      const poll = async () => {
        for (let i = 0; i < 300; i++) {
          await new Promise((r) => setTimeout(r, 2000))
          const s = await pollStatus(run.run_id)
          setStatus(`Run ${run.run_id} — ${s.status}`)
          if (s.status === 'completed' || s.status === 'error') {
            return s.status
          }
        }
        return 'timeout'
      }

      const finalStatus = await poll()
      if (finalStatus === 'completed') {
        setPhase('done')
        navigate(`/runs/${run.run_id}`)
      } else {
        setPhase('error')
        setError(`Run ended with status: ${finalStatus}`)
      }
    } catch (err) {
      setPhase('error')
      setError(err instanceof Error ? err.message : 'Unknown error')
    }
  }, [file, genai, model, maxIters, navigate])

  const busy = phase !== 'idle' && phase !== 'error' && phase !== 'done'

  return (
    <div>
      <h1 style={{ marginBottom: '1.5rem' }}>Upload &amp; Run</h1>

      <div className="card">
        <FileUpload onFileSelected={setFile} disabled={busy} />
      </div>

      <div className="card">
        <h3>Options</h3>
        <div className="flex gap-3" style={{ flexWrap: 'wrap', alignItems: 'flex-end' }}>
          <div>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={genai}
                onChange={(e) => setGenai(e.target.checked)}
                disabled={busy}
              />
              Enable GenAI mode
            </label>
          </div>

          {genai && (
            <>
              <div>
                <label>Model</label>
                <select value={model} onChange={(e) => setModel(e.target.value)} disabled={busy}>
                  {MODELS.map((m) => (
                    <option key={m.value} value={m.value}>{m.label}</option>
                  ))}
                </select>
              </div>
              <div>
                <label>Max Iterations</label>
                <input
                  type="number"
                  min={1}
                  max={10}
                  value={maxIters}
                  onChange={(e) => setMaxIters(parseInt(e.target.value) || 2)}
                  disabled={busy}
                  style={{ width: '80px' }}
                />
              </div>
            </>
          )}
        </div>
      </div>

      <button
        className="btn btn-primary"
        onClick={handleRun}
        disabled={!file || busy}
        style={{ marginBottom: '1rem' }}
      >
        {busy ? (
          <>
            <span className="spinner" style={{ marginRight: '0.5rem', verticalAlign: 'middle' }} />
            Processing...
          </>
        ) : (
          'Upload & Run'
        )}
      </button>

      {status && phase === 'polling' && (
        <div className="card">
          <p>{status}</p>
          <div className="progress-bar">
            <div className="fill" style={{ width: '100%' }} />
          </div>
        </div>
      )}

      {error && (
        <div className="card" style={{ borderColor: 'var(--danger)' }}>
          <p style={{ color: 'var(--danger)' }}>{error}</p>
          <button className="btn btn-secondary mt-1" onClick={() => { setPhase('idle'); setError('') }}>
            Try Again
          </button>
        </div>
      )}

      {runId && phase === 'error' && (
        <button className="btn btn-secondary mt-1" onClick={() => navigate(`/runs/${runId}`)}>
          View Run (may have partial results)
        </button>
      )}
    </div>
  )
}
