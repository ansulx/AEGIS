# Unlock GitHub Profile Achievements with AEGIS

This repo is set up so you (and contributors) can earn **GitHub profile achievement badges** by participating. Achievements appear on your GitHub profile and showcase your activity.

---

## Quick reference

| Achievement | How to unlock | How this repo helps |
|-------------|----------------|----------------------|
| **Quickdraw** | Close an issue or PR within 5 minutes of opening it | Use the [Quickdraw issue template](.github/ISSUE_TEMPLATE/quickdraw.md) |
| **Pull Shark** | Merge 2+ pull requests | Open small PRs (typo, doc, example); we welcome them |
| **YOLO** | Merge a PR without code review | Merge your own PR without requiring review (repo setting) |
| **Starstruck** | Have a repo reach 16+ stars | Share AEGIS; consider starring if it helps you |
| **Pair Extraordinaire** | Co-author 1+ merged PRs | Use `Co-authored-by` in commit messages when pairing |
| **Galaxy Brain** | Get 2+ answers accepted in Discussions | Enable Discussions below, then answer Q&A posts |

---

## Step-by-step: Quickdraw

1. Go to [Issues](https://github.com/ansulx/AEGIS/issues) → **New issue**.
2. Choose **"Quickdraw — open & close an issue"** (or use any template).
3. Submit the issue.
4. **Within 5 minutes**, click **Close issue**.
5. Achievement unlocks on your profile shortly after.

---

## Step-by-step: Pull Shark (merge 2 PRs)

1. **Fork** this repo (top-right **Fork**).
2. Clone your fork, create a branch:  
   `git checkout -b patch-1`
3. Make a small change (e.g. fix a typo in README, add your name to Acknowledgments).
4. Commit and push:  
   `git add . && git commit -m "docs: small improvement" && git push origin patch-1`
5. On GitHub, open **Pull request** from your fork’s branch to `main`.
6. **Merge** the PR (you or a maintainer). Repeat once more for a second merge.
7. After 2 merged PRs, **Pull Shark** unlocks.

---

## Step-by-step: YOLO (merge without review)

1. Same as Pull Shark: fork, branch, small edit, push, open PR.
2. In **this repo**: **Settings** → **General** → **Pull Requests** → turn off "Require a pull request before merging" (or allow merging without approvals for your branch).
3. Open a PR from your fork and click **Merge pull request** without requesting or waiting for review.
4. **YOLO** unlocks.

---

## Step-by-step: Starstruck (16+ stars)

1. Share the repo (e.g. in paper, Twitter, conference, lab).
2. Add a clear **About** description and **Topics** on the repo (clinical-ai, ai-safety, medical-ai, miccai, etc.).
3. In README we ask: "If AEGIS helped you, consider giving it a ⭐."
4. When the repo reaches **16 stars**, **Starstruck** unlocks (you can star your own repo for testing; higher tiers at 128, 512, 4096 stars).

---

## Step-by-step: Pair Extraordinaire (co-authored PRs)

1. When two people work on a commit, add at the bottom of the commit message:
   ```
   Co-authored-by: Other Name <other@email.com>
   ```
2. Merge the PR. When the co-author has at least one such merged PR, **Pair Extraordinaire** unlocks.

---

## Step-by-step: Galaxy Brain (2+ accepted answers)

1. **Enable Discussions** on this repo:  
   **Settings** → **General** → **Features** → check **Discussions**.
2. In the **Discussions** tab, use **Q&A** (or similar) so that the question author can mark an answer as **Accepted**.
3. Answer 2 questions and have your answers marked **Accepted**.
4. **Galaxy Brain** unlocks (note: not all Discussion types award it; Q&A with "Accepted answer" does).

---

## Recommended repo settings (maintainer)

- **Settings → General → Features**: Enable **Discussions** (for Galaxy Brain).
- **Settings → Pull Requests**: Optional — allow merge without required review so you can get YOLO or merge contributor PRs quickly.
- **About (right sidebar)**: Set description and topics (e.g. `clinical-ai`, `ai-safety`, `medical-ai`, `miccai`, `guardrails`).
- **Labels**: Optional labels to add under **Issues → Labels**: `good first issue`, `help wanted`, `Quickdraw`.

---

## Links

- [GitHub’s own achievements overview](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-github-profile/managing-contribution-settings-on-your-profile/about-achievements)
- [Community guide to achievements](https://githubachievements.com/)
