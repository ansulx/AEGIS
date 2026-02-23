# Push AEGIS to GitHub

Follow these steps to publish this repository to GitHub.

## 1. Create the repository on GitHub

1. Go to [https://github.com/new](https://github.com/new).
2. **Repository name:** `AEGIS`
3. **Description:** `A protective shield against silent AI failures in clinical use`
4. Choose **Public**.
5. Do **not** initialize with a README, .gitignore, or license (we already have them).
6. Click **Create repository**.

## 2. Update placeholder username

Replace `ansulx` with your GitHub username in:

- `README.md` (all badge and repo links)
- `CITATION.cff` (repository-code and url)

Quick replace (run from repo root):

```bash
# Replace ansulx with your actual GitHub username, e.g. "johndoe"
sed -i '' 's/ansulx/your-github-username/g' README.md CITATION.cff CONTRIBUTING.md
```

Then commit the change:

```bash
git add README.md CITATION.cff CONTRIBUTING.md
git commit -m "docs: set repository URLs for GitHub"
```

## 3. Add remote and push

From the project root (`MICCAI 2026`):

```bash
git remote add origin https://github.com/ansulx/AEGIS.git
git branch -M main
git push -u origin main
```

Use your actual GitHub username in the `origin` URL. If you use SSH:

```bash
git remote add origin git@github.com:ansulx/AEGIS.git
git push -u origin main
```

## 4. Optional: add topics and link paper

On the GitHub repo page:

- **About** → edit → add **Topics**, e.g. `clinical-ai`, `ai-safety`, `medical-ai`, `miccai`, `guardrails`, `reproducibility`.
- Add **Website** or **DOI** if you have a paper or project page.

After the first push, consider connecting the repo to **Zenodo** (GitHub integration) to get a DOI and improve citability.

---

## 5. Optional: repo settings

- **Settings** → **General** → **Features** → enable **Discussions** if you want a Q&A or discussion area.
- **Settings** → **Pull Requests** → optionally allow merging without required review (useful for small doc/typo PRs).
