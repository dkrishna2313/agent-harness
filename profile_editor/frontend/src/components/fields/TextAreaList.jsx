import React from 'react'

// Renders a list of strings as plain text inputs (one per row) — good for regex patterns
export default function TextAreaList({ value = [], onChange }) {
  const items = Array.isArray(value) ? value : []

  const update = (i, v) => {
    const next = [...items]
    next[i] = v
    onChange(next)
  }

  const remove = (i) => onChange(items.filter((_, idx) => idx !== i))

  const add = () => onChange([...items, ''])

  return (
    <div className="space-y-1.5">
      {items.map((item, i) => (
        <div key={i} className="flex gap-2">
          <input
            type="text"
            value={item}
            onChange={(e) => update(i, e.target.value)}
            className="flex-1 text-xs font-mono border border-slate-200 rounded px-2.5 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-300"
            placeholder="pattern..."
          />
          <button
            onClick={() => remove(i)}
            className="text-slate-400 hover:text-red-500 text-lg leading-none px-1"
            aria-label="Remove"
          >
            ×
          </button>
        </div>
      ))}
      <button
        onClick={add}
        className="text-sm text-blue-600 hover:underline mt-1"
      >
        + Add pattern
      </button>
    </div>
  )
}
