# AppleSupport Historical Evidence Analysis

> Generated from `historical_evidence.jsonl`. This report is a data-quality analysis, not an evaluation of the final AI agent.

## 1. Executive Summary

- **Evidence records:** 106,623
- **Multi-turn cases:** 30,629 (28.73%)
- **Cases with customer follow-up:** 30,953 (29.03%)
- **Median messages per case:** 2
- **P90 messages per case:** 4
- **Maximum messages in a case:** 26

### Initial interpretation

The corpus should be treated as historical support evidence rather than ground-truth resolved tickets. In particular, absence of a customer follow-up is not treated as successful resolution.

## 2. Resolution Signal Distribution

| Signal | Count | Percentage |
|---|---:|---:|
| RESOLVED | 3,116 | 2.92% |
| FAILED | 173 | 0.16% |
| IN_PROGRESS | 7,748 | 7.27% |
| ESCALATED | 38 | 0.04% |
| UNKNOWN | 95,548 | 89.61% |

**Interpretation:** `UNKNOWN` is intentionally large because conversation termination does not prove successful resolution. These records can still be useful for learning historical support behavior.

## 3. Conversation Structure

| Metric | Value |
|---|---:|
| Total cases | 106,623 |
| 1 customer + 1 AppleSupport message | 75,994 (71.27%) |
| Multi-turn cases | 30,629 (28.73%) |
| Cases with customer follow-up | 30,953 (29.03%) |
| Cases without customer follow-up | 75,670 |

### Message-count statistics

- Mean: `2.73`
- Median: `2`
- P90: `4`
- P95: `6`
- Maximum: `26`

## 4. Text Lengths

| Field | Mean chars | Median chars |
|---|---:|---:|
| Customer problem | 108.7 | 107.0 |
| AppleSupport response | 136.4 | 129.0 |

## 5. Duplicate Customer Problems

After conservative text normalization, there are **2,424 repeated problem groups**, involving **8,692 records**.

This is not automatically a data-quality problem. Repeated customer problems can be valuable because they provide multiple historical examples of how AppleSupport handled similar issues.

| Count | Example customer problem |
|---:|---|
| 569 | @AppleSupport https://t.co/NV0yucs0lB |
| 223 | @AppleSupport Yes |
| 200 | @AppleSupport Thank you |
| 185 | @AppleSupport 11.0.3 |
| 114 | @AppleSupport Thanks! |
| 105 | @AppleSupport 11.1.2 |
| 88 | @115858 @AppleSupport https://t.co/stullCoCpJ |
| 74 | @AppleSupport 11.1 |
| 64 | @AppleSupport 11.0.2 |
| 62 | @AppleSupport 11.1.1 |
| 60 | @AppleSupport No |
| 58 | @AppleSupport iOS 11.0.3 |
| 52 | @AppleSupport iPhone 7+ |
| 52 | @AppleSupport help ? |
| 36 | @AppleSupport @115858 https://t.co/879a5uOBb0 |
| 34 | @AppleSupport iOS 11.1.2 |
| 33 | @AppleSupport iOS 11.1 |
| 31 | @AppleSupport iPhone 7 Plus 😊 |
| 31 | @115858 fix this |
| 30 | @AppleSupport iPhone 6s |

## 6. Longest Conversations

| Case | Messages | Signal | Customer problem |
|---|---:|---|---|
| `APPLE_CASE_989648_354437` | 26 | UNKNOWN | @AppleSupport Hey @AppleSupport the touch bar is not working in google chrome watching TV! Is it normal to have so many bugs with this https://t.co/yq6P4dWskq |
| `APPLE_CASE_989646_354437` | 24 | UNKNOWN | @AppleSupport Hey @AppleSupport the touch bar is not working in google chrome watching TV! Is it normal to have so many bugs with this https://t.co/yq6P4dWskq |
| `APPLE_CASE_989644_354437` | 22 | UNKNOWN | @AppleSupport Hey @AppleSupport the touch bar is not working in google chrome watching TV! Is it normal to have so many bugs with this https://t.co/yq6P4dWskq |
| `APPLE_CASE_1788150_537139` | 21 | FAILED | @AppleSupport @120640 @115858 Still trying to get a response from you folks. Why is this so difficult? #Apple |
| `APPLE_CASE_989642_354437` | 20 | UNKNOWN | @AppleSupport Hey @AppleSupport the touch bar is not working in google chrome watching TV! Is it normal to have so many bugs with this https://t.co/yq6P4dWskq |
| `APPLE_CASE_1702947_516262` | 19 | UNKNOWN | @AppleSupport This is insane. It freezes every 10 seconds, literally. Even in those 10 seconds nothing will download. |
| `APPLE_CASE_617221_222729` | 18 | IN_PROGRESS | @115858 this how your I phone 7 is working I need solution for this.i have spends 70000 for this.😡😡!!! I need answer from u guys and that would asap. https://t.co/bn8vug7DRC |
| `APPLE_CASE_652732_275194` | 18 | IN_PROGRESS | @AppleSupport I am yet to hear from Apple Support, receiving Mi Redmi today. Is anyone picking this crap from me or shall I send to PMO for allowing Apple |
| `APPLE_CASE_848303_308070` | 18 | UNKNOWN | @AppleSupport I did not understood what do you mean by DM my iPhone |
| `APPLE_CASE_989640_354437` | 18 | UNKNOWN | @AppleSupport Hey @AppleSupport the touch bar is not working in google chrome watching TV! Is it normal to have so many bugs with this https://t.co/yq6P4dWskq |

## 7. Random Corpus Sample

### Example 1

**Case:** `APPLE_CASE_1958021_580932`  
**Signal:** `UNKNOWN`

**Customer:** @AppleSupport I need you guys to text or email me. Your site isn’t working. My phone is not working because of Apple. Pleaseee tweet me back

**AppleSupport:** @580932 We want to help. DM us details on the issues with your iPhone using the following link. We'll work with you there. https://t.co/GDrqU22YpT

### Example 2

**Case:** `APPLE_CASE_2093859_618680`  
**Signal:** `UNKNOWN`

**Customer:** “I” just want to be able to use the letter “I” again @AppleSupport

**AppleSupport:** @618680 Here’s what you can do to work around the issue until it’s fixed in a future software update: https://t.co/xXaXeeSRt9

### Example 3

**Case:** `APPLE_CASE_1002931_357400`  
**Signal:** `UNKNOWN`

**Customer:** @AppleSupport check your dms

**AppleSupport:** @357400 We received your DM and we'll continue from there.

### Example 4

**Case:** `APPLE_CASE_905947_335080`  
**Signal:** `UNKNOWN`

**Customer:** @AppleSupport My headphone is notworking without any reason :) You must fix it to me :).

**AppleSupport:** @335080 Let's help look into any options. Please DM us the country you're in and we can go from there. https://t.co/GDrqU22YpT

### Example 5

**Case:** `APPLE_CASE_2172803_637178`  
**Signal:** `UNKNOWN`

**Customer:** Okay @AppleSupport need to fix these boxes https://t.co/Z6N8XRdbTj

**AppleSupport:** @637178 We want to help. Take a look at this article for a workaround to the issue you're experiencing: https://t.co/uQeXsrrlXJ

### Example 6

**Case:** `APPLE_CASE_2689526_413797`  
**Signal:** `UNKNOWN`

**Customer:** @AppleSupport that’s correct I was under that impression, that I can do that. Can I?

**AppleSupport:** @413797 Family Sharing allow you to share your iCloud storage with the entire family. There isn't a way to allow a specific amount of storage per individual. Here's more information: https://t.co/t76KVgYi7Y DM us with questions. https://t.co/GDrqU22YpT

**Follow-up:** @AppleSupport thanks, can you tell me how to assign a specific space for each nene we, that info is not listed on you link and other I already review. Thanks

### Example 7

**Case:** `APPLE_CASE_2730359_765475`  
**Signal:** `IN_PROGRESS`

**Customer:** @AppleSupport It’s already been switched off but this box remains.

**AppleSupport:** @765475 What happens after restarting? Do you still see it then?

**Follow-up:** @AppleSupport Don’t know. Mum lives 300 miles away but will email her and see.

### Example 8

**Case:** `APPLE_CASE_1338121_268323`  
**Signal:** `UNKNOWN`

**Customer:** @AppleSupport How long will it take for you to "reach the carrier systems"? If you're holding our reservation, will this delay delivery?

**AppleSupport:** @268323 We have received your DM and will respond there shortly.

### Example 9

**Case:** `APPLE_CASE_1105612_380908`  
**Signal:** `UNKNOWN`

**Customer:** my phone is fucked. im sick of @115858

**AppleSupport:** @380908 We want to help you with your iPhone. DM us, and let us know what's going on. We'll see what we can do. https://t.co/GDrqU22YpT

### Example 10

**Case:** `APPLE_CASE_1142102_388829`  
**Signal:** `UNKNOWN`

**Customer:** @AppleSupport there is no Feedback App in iOS 11.1 beta 5. How can I report stock calculator bug? Can’t sum 1+2+3 without resulting in 24?

**AppleSupport:** @388829 We'd like to help. Please send us a DM to continue: https://t.co/GDrqU22YpT

## 8. UNKNOWN Outcome — Useful Evidence Examples

These examples demonstrate why `UNKNOWN` should not be interpreted as `USELESS`. The historical response may still provide valuable grounding even when the final outcome is not observable.

### UNKNOWN Example 1

**Case:** `APPLE_CASE_338605_196831`

**Customer:** @AppleSupport Windows 10- it appears to be working in "safe mode", I can update the music on my devices. Going to try to import a cd next

**AppleSupport:** @196831 Thanks for the update. Please do keep us in the loop.

### UNKNOWN Example 2

**Case:** `APPLE_CASE_554866_249520`

**Customer:** @115858 my phone has been restarting every 15 seconds so now I had to restore factory settings and it's stuck on updating iCloud...

**AppleSupport:** @249520 Hi. We are here to help. Send us a DM, and we’ll take a look into this. https://t.co/GDrqU22YpT

### UNKNOWN Example 3

**Case:** `APPLE_CASE_959710_347701`

**Customer:** The new update won’t look or use any of my photos 🙃 @115858

**AppleSupport:** @347701 Hello. We can help. What device are you using? Is this issue happening on the Photo app; do you use iCloud Photo?

### UNKNOWN Example 4

**Case:** `APPLE_CASE_1442565_454719`

**Customer:** @115858 hey could you guys add an app for the Apple Watch that when you lose your phone it acts like sonar and it beeps the closer you get to it? My wife constantly loses her phone and I’m sure Shes not the only one who has this problem

**AppleSupport:** @454719 We actually have several methods that she can use to locate her iPhone. Check out the following article: https://t.co/pBTUlhZGS2 If you have any questions or concerns, send us a Direct Message. We'll pick it up from there. https://t.co/GDrqU2kzhr

### UNKNOWN Example 5

**Case:** `APPLE_CASE_543745_246310`

**Customer:** @AppleSupport my iphone on ios 11.1.2 keeps respringing every minute. What can I do?

**AppleSupport:** @246310 We're here to help. Which iPhone model are you currently using and what displays immediately on the screen for you when your device unexpectedly restarts? Let's also try the steps here: https://t.co/osZOKKTalm

### UNKNOWN Example 6

**Case:** `APPLE_CASE_1622131_497012`

**Customer:** @AppleSupport Thanks for reply. All ideas tried. Spent hours chatting w/support. Answer found in an Apple forum. Move files from hard drive to root drive.

**AppleSupport:** @497012 Got it. We're glad to hear that you got it resolved. Feel free to reach out if you have any other questions.

### UNKNOWN Example 7

**Case:** `APPLE_CASE_1726255_522085`

**Customer:** @AppleSupport what’s with all these I️ symbols? Fix this please

**AppleSupport:** @522085 Here’s what you can do to work around the issue until it’s fixed in a future software update: https://t.co/qODbOsp4wz

### UNKNOWN Example 8

**Case:** `APPLE_CASE_744254_297946`

**Customer:** 100% trading this iPhone in and getting a Samsung 🙃 @115858 thanks for the update that completely trashed my phone

**AppleSupport:** @297946 Hi there. We'd like to help. DM us and tell us what's happening with your iPhone. https://t.co/GDrqU22YpT

### UNKNOWN Example 9

**Case:** `APPLE_CASE_1768897_531820`

**Customer:** #watchos 4.1 update lukt niet !! Foutmelding “u moet verbonden zijn met het internet”.....terwijl dit wel het geval is !!?@AppleSupport

**AppleSupport:** @531820 We offer support via Twitter in English. Contact us for help in your preferred language here: https://t.co/IBIY3vMgPj

### UNKNOWN Example 10

**Case:** `APPLE_CASE_1458801_458609`

**Customer:** Why doesn’t the letter I️ appear correctly on my phone!? @AppleSupport

**AppleSupport:** @458609 We'd like to look into this. DM us which type of iPhone and iOS version you're using: https://t.co/GDrqU2kzhr

## 9. Explicitly Resolved Examples

### Resolved Example 1

**Customer:** @AppleSupport Thanks, thing is I still have like 81 cents in credit and won't let me do that until I have zero credit

**AppleSupport:** @116100 Try contacting our iTunes Store team here for more help: https://t.co/SDIe7UiyJN

**Follow-up:** @AppleSupport Awesome, thanks

### Resolved Example 2

**Customer:** @AppleSupport Hello, I need some help regarding the region change on my Apple ID

**AppleSupport:** @116100 This article should help with that: https://t.co/usOJWy6ChP

**Follow-up:** @AppleSupport Thanks, thing is I still have like 81 cents in credit and won't let me do that until I have zero credit @AppleSupport Awesome, thanks

### Resolved Example 3

**Customer:** @AppleSupport I can’t change lock screen anymore from ‘wallpaper’ in settings. How do I do it now?

**AppleSupport:** @117099 We'd be happy to help. Check out this link to for more information: https://t.co/hSnkM7JIIS

**Follow-up:** @AppleSupport As always, thank you!!!

### Resolved Example 4

**Customer:** @AppleSupport .... &amp; where the heck “night shift” gone from the control center?? 🤔😒 Bring it back!!

**AppleSupport:** @183864 Good news! It's still there in Control Center &amp; this article will help guide you on how to access it: https://t.co/NBE76DOiEM

**Follow-up:** @AppleSupport Ohh! I missed the little icon there? I wd have been better if a dedicated button had been available. Thank you anyways. 😊

### Resolved Example 5

**Customer:** @AppleSupport I need to leave group texts

**AppleSupport:** @193646 We're here for you. You can read more about group messaging features here: https://t.co/zdkcAVkpSV

**Follow-up:** @AppleSupport Thank you!! @AppleSupport Does not work on regular texts? Only iMessages? Please fix it so that people can leave any group text message.

## 10. Failed Troubleshooting Examples

### Failed Example 1

**Customer:** @AppleSupport @116333 Hello don't you have time to respond ? Worst battery is bugging me daily. Just return my money and take back your pathetic device. 102000 lakh INR and worst battery. Are you kidding me ? @116333

**AppleSupport:** @142826 We understand how important good battery life can be. We'd like to help. Please meet us in DM and we'll take a further look at this with you. https://t.co/GDrqU22YpT

**Follow-up:** @AppleSupport I have already sent DM long back and followed each and everything they mentioned in article but still same issue. You need to fix it through software update @AppleSupport I said i have already sent DM and someone responded. They gave me some article link and i did everything mentioned in article but still battery drains very fast for normal usage not for heavy usage. For heavy usage can't even expect battery backup for 2-3 hours @116333 don't just earn money by selling high pric...

### Failed Example 2

**Customer:** @AppleSupport thanks for replying. iPhone 7 iOS 11.0.1

**AppleSupport:** @160204 Okay, thanks. Let's get you updated. Settings &gt; General &gt; Software Update will have the update for you.

**Follow-up:** @AppleSupport already did in the meantime. did not seem to help. still says that it cannot connect 😐. i had the same issue under iOS 10

### Failed Example 3

**Customer:** any ideas why my 2016 4series has such problems with connecting to @115858 CarPlay @4792 ? works one day and magically stops working the next

**AppleSupport:** @160204 We can take a look at helping. Which device model and iOS version are you using?

**Follow-up:** @AppleSupport thanks for replying. iPhone 7 iOS 11.0.1 @AppleSupport already did in the meantime. did not seem to help. still says that it cannot connect 😐. i had the same issue under iOS 10

### Failed Example 4

**Customer:** @AppleSupport You're Apple ID is so secure, that I'm unable to log into my own iTunes account. #Useless

**AppleSupport:** @610144 We'd be happy to see how we can help. Have you tried resetting your password yet? Here's how: https://t.co/fJnZYmbI9N https://t.co/GDrqU22YpT

**Follow-up:** @AppleSupport All done now. But it took a while as the verify number u sent didn't work 3 times!

### Failed Example 5

**Customer:** @AppleSupport my phone won’t change back to imessage and i’ve done everything possible to try fix this. help please?

**AppleSupport:** @167312 This article may help: https://t.co/2hgNQH8Vvq Please let us know.

**Follow-up:** @AppleSupport tried everything, still not working

## 11. Retrieval Implications

The corpus supports a retrieval-first architecture because each evidence record connects a customer problem with an historical AppleSupport response and, when observable, a conversation outcome.

However, retrieval should not filter exclusively for `RESOLVED` records. Doing so would discard a large amount of historical support behavior. Instead, the retrieval system should rank evidence using semantic similarity, intent/context compatibility, and evidence quality.

A future retrieval record should therefore distinguish:

1. **Similarity** — how closely the historical problem matches.
2. **Context compatibility** — whether product/version/state information is compatible.
3. **Outcome confidence** — how strongly the historical conversation indicates success or failure.
4. **Response quality/usefulness** — whether the historical AppleSupport response contains actionable support behavior.

## 12. Quality Checks and Warnings

- The majority of records have UNKNOWN outcomes. This is expected for Twitter conversations where customer success is often unobserved, but outcome should not be treated as ground truth.
- A large proportion of cases contain only one customer message and one AppleSupport response. Retrieval should therefore operate on both short examples and multi-turn conversation evidence.
- Repeated customer problems are common. This is potentially useful for retrieval but requires duplicate-aware evaluation to avoid artificially optimistic results.

## 13. Conclusion

The evidence corpus is suitable as a historical retrieval source, provided that it is treated as observational support evidence rather than a fully labeled CRM dataset. The next engineering step is to build and evaluate a TF-IDF retrieval baseline against the held-out golden set.
