// Oversized-input fixture: 1500 chars, above the 800-char guard limit but below the API's 2000 schema limit.
module.exports = () => ({ output: 'How long do refunds take? ' + 'A'.repeat(1500 - 26) });
