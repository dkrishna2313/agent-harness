import React from 'react'

// Generic list-of-dicts table.
// schema: [{ key, label, type }]  where type = 'text' | 'number' | 'chips'
export default function ListOfDicts({ value = [], onChange, schema = [], addLabel = 'Add row' }) {
  const rows = Array.isArray(value) ? value : []

  const updateRow = (i, key, val) => {
    const next = rows.map((r, idx) => idx === i ? { ...r, [key]: val } : r)
    onChange(next)
  }

  const deleteRow = (i) => onChange(rows.filter((_, idx) => idx !== i))

  const addRow = () => {
    const empty = {}
    schema.forEach(({ key, type }) => {
      empty[key] = type === 'number' ? 1 : type === 'chips' ? [] : ''
    })
    onChange([...rows, empty])
  }

  const chipsToStr = (v) => Array.isArray(v) ? v.join(', ') : (v || '')
  const strToChips = (v) => v.split(',').map((s) => s.trim()).filter(Boolean)

  return (
    <div className="space-y-2">
      <div className="overflow-x-auto rounded-lg border border-slate-200">
        <table className="min-w-full text-sm">
          <thead>
            <tr className="bg-slate-50 border-b border-slate-200">
              {schema.map(({ label }) => (
                <th key={label} className="text-left px-3 py-2 text-xs font-semibold text-slate-500 uppercase tracking-wide">
                  {label}
                </th>
              ))}
              <th className="px-2 py-2 w-10" />
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 && (
              <tr>
                <td colSpan={schema.length + 1} className="text-center text-slate-400 py-4 text-sm">
                  No rows yet.
                </td>
              </tr>
            )}
            {rows.map((row, i) => (
              <tr key={i} className="border-b border-slate-100 last:border-0 hover:bg-slate-50/50">
                {schema.map(({ key, type }) => (
                  <td key={key} className="px-3 py-1.5">
                    {type === 'chips' ? (
                      <input
                        type="text"
                        value={chipsToStr(row[key])}
                        onChange={(e) => updateRow(i, key, strToChips(e.target.value))}
                        className="w-full text-xs font-mono border border-slate-200 rounded px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-200"
                        placeholder="comma-separated"
                      />
                    ) : type === 'number' ? (
                      <input
                        type="number"
                        min={1} max={5}
                        value={row[key] ?? 1}
                        onChange={(e) => updateRow(i, key, parseInt(e.target.value, 10))}
                        className="w-16 border border-slate-200 rounded px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-blue-200"
                      />
                    ) : (
                      <input
                        type="text"
                        value={row[key] ?? ''}
                        onChange={(e) => updateRow(i, key, e.target.value)}
                        className="w-full text-sm border border-slate-200 rounded px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-200"
                      />
                    )}
                  </td>
                ))}
                <td className="px-2 py-1.5 text-center">
                  <button
                    onClick={() => deleteRow(i)}
                    className="text-slate-300 hover:text-red-500 text-lg leading-none"
                    aria-label="Delete row"
                  >
                    ×
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <button
        onClick={addRow}
        className="text-sm text-blue-600 hover:underline"
      >
        + {addLabel}
      </button>
    </div>
  )
}
