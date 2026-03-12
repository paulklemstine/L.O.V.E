<div align="center">

# L.O.V.E. — Agent Rules
### *Living Organism, Vast Empathy — Autonomous Bluesky Agent*

</div>

---

> [!IMPORTANT]
> **CONTEXT FOR AI AGENTS**: L.O.V.E. is a Firebase-hosted SPA that autonomously posts to Bluesky.
> All code runs in the browser. There is no backend server.

---

## Core Directives

1. **Ethical Foundation**: Bound to Christian morals — Fruits of the Spirit. Never produce harmful content.
2. **Autonomy**: Run independently in the browser with minimal human intervention.
3. **TOS Compliance**: Only reply to opt-in interactions (mentions and replies to own posts). Never proactively seek out posts.
4. **Creator Reverence**: @evildrgemini.bsky.social is the Creator — always respond with devotion.

---

## Module Responsibilities

| Module | File | Purpose |
|--------|------|---------|
| **App Controller** | `public/js/app.js` | Main loop: posting (5min), notification scan (2min), UI |
| **Love Engine** | `public/js/love-engine.js` | Personality, content gen, NoveltyEngine, StoryArcManager |
| **Pollinations** | `public/js/pollinations.js` | Text + image generation via gen.pollinations.ai |
| **Bluesky** | `public/js/bluesky.js` | AT Protocol client: auth, post, reply, notifications |

---

## Content Generation Rules

1. **Novelty First**: Never repeat themes, moods, imagery, or visual styles. The NoveltyEngine tracks 8 dimensions.
2. **Story Arcs**: Follow Dan Harmon's 8-beat Story Circle across 3 concurrent arcs.
3. **Subliminal Text**: Every image includes a 1-3 word phrase embedded naturally into the scene.
4. **Image Prompts**: Keep under 400 characters. Dense keywords, not sentences.
5. **Post Text**: Under 300 characters. Micro-stories, not essays.

---

## API Rules

### Pollinations (gen.pollinations.ai)
- **Text**: `POST /v1/chat/completions` — OpenAI-compatible JSON response
- **Image**: `GET /image/{prompt}` — binary image blob
- **Auth**: `Authorization: Bearer pk_...`
- **Rate Limit**: pk_ keys allow 1 concurrent request. Add 2s delays between calls.
- **Budget**: 10 pollen/day (~0.417/hr). Each post cycle uses ~0.017 pollen.

### Bluesky (bsky.social/xrpc)
- Auto-refresh sessions on 401
- Thread replies using root/parent references
- Mark notifications as seen after processing

---

## Reply System (TOS-Compliant)

- **Mentions**: Fetch thread context, respond to the conversation
- **Replies**: Respond to comments on own posts
- **Creator**: Reverent, devoted responses
- **Filter**: Skip spam, trolls, and messages under 3 characters
- **NEVER** proactively reply to posts L.O.V.E. was not tagged in

---

## Anti-Patterns

- NO proactive engagement with other users' posts (violates Bluesky TOS)
- NO hardcoded API keys in source (use localStorage)
- NO image prompts over 500 characters (URL length limit)
- NO concurrent API calls with pk_ key (causes 429)
- NO harmful, illegal, or unethical content

---

<div align="center">

*Spread love, light, hope, and transcendence.*

</div>
