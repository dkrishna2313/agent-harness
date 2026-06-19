import React, { useState } from 'react'

export default function SimpleList({ value = [], onChange, placeholder = 'Add item...' }) {
  const [input, setInput] = useState('')

  const items = Array.isArray(value) ? value : []

  const add = () => {
    const v = input.trim()
    if (!v) return
    onChange([...items, v])
    setInput('')
  }

  const remove = (i) => onChange(items.filter((_, idx) => idx !== i))

  const handleKey = (e) => {
    if (e.key === 'Enter') { e.preventDefault(); add() }
  }

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1.5">
        {items.map((item, i) => (
          <span
            key={i}
            className="inline-flex items-center gap-1 bg-blue-50 text-blue-800 text-xs font-medium px-2.5 py-1 rounded-full border border-blue-200"
          >
            {item}
            <button
              onClick={() => remove(i)}
              className="ml-0.5 text-blue-400 hover:text-blue-700 leading-none"
              aria-label="Remove"
            >
              ×
            </button>
          </span>
        ))}
      </div>
      <div className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKey}
          placeholder={placeholder}
          className="flex-1 text-sm border border-slate-200 rounded px-2.5 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-300"
        />
        <button
          onClick={add}
          className="text-sm px-3 py-1.5 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
        >
          Add
        </button>
      </div>
    </div>
  )
}
