\# Hiver Support Agent



\## 1. Overview



This project implements an AI-assisted customer-support agent for AppleSupport conversations from the Twitter Customer Support dataset.



The agent:



1\. Reconstructs customer-support cases from tweet relationships.

2\. Classifies the customer's primary intent.

3\. Extracts useful entities, symptoms, claims, and risk signals.

4\. Retrieves similar historical cases.

5\. Validates whether retrieved evidence is safe to reuse.

6\. Automatically handles, asks for clarification, or escalates the case.



The system is deliberately conservative: similarity alone is not treated as proof that a historical response is appropriate.



\## 2. Dataset and Case Reconstruction



The selected brand was AppleSupport.



| Metric | Value |

|---|---:|

| Total tweets | 2,811,774 |

| AppleSupport tweets | 106,860 |

| Reconstructed cases | 106,623 |

| Reconstructed messages | 290,796 |

| Historical resolved cases | 3,116 |

| Historical unknown outcomes | 95,548 |

| Multi-turn cases | 30,629 |



A case is reconstructed by following the tweet-response graph around AppleSupport replies and grouping messages belonging to the same customer-support interaction.



An important limitation is that the absence of a customer follow-up does not prove resolution. Therefore, unresolved or unknown outcomes are retained rather than being incorrectly labelled as resolved.



\## 3. Architecture



```text

Customer message

&#x20;     |

&#x20;     v

Case reconstruction

&#x20;     |

&#x20;     v

Issue analysis

&#x20;     |

&#x20;     +--> intent

&#x20;     +--> entities

&#x20;     +--> symptoms

&#x20;     +--> conversation state

&#x20;     +--> risk flags

&#x20;     |

&#x20;     v

Historical evidence retrieval

&#x20;     |

&#x20;     v

Evidence-quality assessment

&#x20;     |

&#x20;     v

Deterministic policy

&#x20;     |

&#x20;     +--> AUTO\_HANDLE

&#x20;     +--> CLARIFY

&#x20;     +--> ESCALATE

&#x20;     |

&#x20;     v

Draft response or escalation reason

