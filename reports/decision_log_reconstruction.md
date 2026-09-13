# Decision Log — Conversation Reconstruction

## Decision 017 — Model support interactions as customer/AppleSupport branches, not whole public roots

**Problem:** TWCS is a Twitter reply graph, not a CRM export. A root thread can contain a company/marketing tweet and unrelated customer branches. Treating the entire root as one support case contaminates historical evidence.

**Initial approach:** Reconstruct the complete descendant tree from a conversation root.

**Observed failure:** A validation sample included unrelated AppleSupport messages and, in some roots, multiple customer participants. For example, a root could be an Apple marketing tweet followed by a customer's support complaint; the marketing tweet is not part of the support interaction.

**Final approach:** Anchor a support interaction on a customer tweet directly answered by AppleSupport. Traverse descendants while retaining only that customer and AppleSupport. Preserve the original TWCS root tweet ID only as provenance.

**Why:** This keeps the customer → AppleSupport → customer follow-up chain while preventing unrelated public-thread participants from becoming evidence for the same support case.

**Trade-off:** Some public-thread context may be intentionally excluded. This is preferable to contaminating retrieval with unrelated customers or support responses.

**Implication:** The derived `case_id` is an internal support-interaction identifier, not an original TWCS case ID.
