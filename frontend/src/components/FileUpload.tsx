import { useCallback, useState, useRef } from 'react'

interface Props {
  onFileSelected: (file: File) => void
  disabled?: boolean
}

export default function FileUpload({ onFileSelected, disabled }: Props) {
  const [dragActive, setDragActive] = useState(false)
  const [fileName, setFileName] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = useCallback(
    (file: File) => {
      const ext = file.name.toLowerCase()
      if (!ext.endsWith('.dcm') && !ext.endsWith('.dicom')) {
        alert('Please select a DICOM file (.dcm or .dicom)')
        return
      }
      setFileName(file.name)
      onFileSelected(file)
    },
    [onFileSelected],
  )

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setDragActive(false)
      if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0])
      }
    },
    [handleFile],
  )

  return (
    <div
      className={`drop-zone ${dragActive ? 'active' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
      onDragLeave={() => setDragActive(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".dcm,.dicom"
        style={{ display: 'none' }}
        disabled={disabled}
        onChange={(e) => {
          if (e.target.files?.[0]) handleFile(e.target.files[0])
        }}
      />
      {fileName ? (
        <p style={{ color: 'var(--accent)' }}>&#128196; {fileName}</p>
      ) : (
        <>
          <p style={{ fontSize: '2rem' }}>&#128203;</p>
          <p>Drop a DICOM file here or click to browse</p>
          <p style={{ fontSize: '0.8rem' }}>.dcm / .dicom</p>
        </>
      )}
    </div>
  )
}
