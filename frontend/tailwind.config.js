/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        dark: {
          900: '#0b0f19',
          800: '#111827',
          700: '#1f2937',
          600: '#374151',
        },
        boat: {
          blue: '#2563eb',
          cyan: '#06b6d4',
          hit: '#10b981',
          miss: '#374151',
          pool: '#f59e0b',
        }
      }
    },
  },
  plugins: [],
}
