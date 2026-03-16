/**
 * Sanitize HTML to prevent XSS attacks.
 *
 * Uses an allowlist approach: only known-safe tags and attributes survive.
 * Everything else is stripped or entity-escaped.
 */

// Tags that are safe to render (block + inline formatting only)
const ALLOWED_TAGS = new Set([
  'p', 'br', 'hr',
  'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
  'ul', 'ol', 'li',
  'blockquote', 'pre', 'code',
  'strong', 'b', 'em', 'i', 'u', 's', 'del', 'sub', 'sup',
  'a', 'span', 'div',
  'table', 'thead', 'tbody', 'tr', 'th', 'td',
])

// Attributes that are safe on any tag
const ALLOWED_ATTRS = new Set([
  'class', 'id', 'data-level',
])

// Attributes that are safe only on specific tags
const TAG_ATTRS = {
  a: new Set(['href', 'title', 'target', 'rel']),
  td: new Set(['colspan', 'rowspan']),
  th: new Set(['colspan', 'rowspan']),
}

// Protocols allowed in href
const SAFE_URL_RE = /^(?:https?:|mailto:|#|\/)/i

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

/**
 * Sanitize an HTML string using an allowlist of tags and attributes.
 *
 * @param {string} html - Raw HTML string
 * @returns {string} Sanitized HTML
 */
export function sanitizeHtml(html) {
  if (!html) return ''

  // Use the browser's DOM parser to handle all edge cases
  // (nested tags, malformed HTML, encoding tricks, etc.)
  if (typeof DOMParser !== 'undefined') {
    return sanitizeViaDom(html)
  }

  // Fallback: aggressive strip for SSR / non-browser environments
  return stripAllTags(html)
}

function sanitizeViaDom(html) {
  const doc = new DOMParser().parseFromString(html, 'text/html')
  return sanitizeNode(doc.body)
}

function sanitizeNode(node) {
  let out = ''
  for (const child of node.childNodes) {
    if (child.nodeType === 3 /* TEXT */) {
      out += escapeHtml(child.textContent)
    } else if (child.nodeType === 1 /* ELEMENT */) {
      const tag = child.tagName.toLowerCase()
      if (ALLOWED_TAGS.has(tag)) {
        out += `<${tag}`
        // Filter attributes
        for (const attr of child.attributes) {
          const name = attr.name.toLowerCase()
          // Skip event handlers and dangerous attributes
          if (name.startsWith('on')) continue
          if (!ALLOWED_ATTRS.has(name) && !(TAG_ATTRS[tag]?.has(name))) continue
          let val = attr.value
          // Validate URLs in href/src
          if (name === 'href' && !SAFE_URL_RE.test(val.trim())) continue
          out += ` ${name}="${escapeHtml(val)}"`
        }
        // Force rel="noopener noreferrer" on links
        if (tag === 'a') {
          out += ' rel="noopener noreferrer"'
        }
        out += '>'
        out += sanitizeNode(child)
        // Void elements don't need closing tags
        if (!['br', 'hr'].includes(tag)) {
          out += `</${tag}>`
        }
      } else {
        // Disallowed tag: recurse into children (keep text, drop tag)
        out += sanitizeNode(child)
      }
    }
  }
  return out
}

function stripAllTags(html) {
  // Aggressive fallback: remove all HTML tags
  return html.replace(/<[^>]*>/g, '')
}
