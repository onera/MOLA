# MOLA CI/CD Variables Configuration

This document describes the GitLab CI/CD variables required for the automated deployment pipeline.

## Required Variables

All variables below should be configured as **Protected** and **Masked** in GitLab:
- Go to GitLab Project → Settings → CI/CD → Variables
- Click "Add variable" for each entry
- Check both "Protected" and "Masked" options

### 1. `MOLA_DEPLOY_SSH_KEY` (File Variable)

**Purpose:** SSH private key for deploying to `/stck/mola/` (GitLab File variable)

**Type:** File (not Variable) - GitLab will store the key content as a temporary file and provide its path

**How to generate:**

1. **As `mola` user on JUNO, generate SSH key pair:**
   ```bash
   su - mola
   mkdir -p ~/.ssh
   chmod 700 ~/.ssh
   ssh-keygen -t ed25519 -C "gitlab-ci-deployment" -f ~/.ssh/id_gitlab_ci -N ""
   ```

2. **Add public key to authorized_keys:**
   ```bash
   cat ~/.ssh/id_gitlab_ci.pub >> ~/.ssh/authorized_keys
   chmod 600 ~/.ssh/authorized_keys
   ```

3. **Add localhost to known_hosts:**
   ```bash
   ssh-keyscan -H localhost >> ~/.ssh/known_hosts
   chmod 644 ~/.ssh/known_hosts
   ```

4. **Test SSH from `lbernard` to `mola@localhost`:**
   ```bash
   exit  # back to lbernard
   ssh -i ~/.ssh/id_gitlab_ci -o IdentitiesOnly=yes mola@localhost "echo SSH_works"
   ```

5. **Copy the private key content:**
   ```bash
   cat ~/.ssh/id_gitlab_ci
   ```
   Copy the entire output (including `-----BEGIN OPENSSH PRIVATE KEY-----` and `-----END OPENSSH PRIVATE KEY-----`).

**How to configure in GitLab:**
1. Go to Settings → CI/CD → Variables
2. Click "Add variable"
3. **Key:** `MOLA_DEPLOY_SSH_KEY`
4. **Type:** Select "File" (not Variable)
5. **Value:** Paste the entire private key content
6. **Protected:** ✅ Yes
7. **Masked:** ✅ Yes
8. Click "Add variable"

**Note:** File variables are more reliable than base64-encoded variables because GitLab handles the file creation directly, avoiding encoding issues.

### 2. `PYPI_API_TOKEN`

**Purpose:** PyPI API token for uploading releases via `twine`

**How to obtain:**
1. Create an account on PyPI (if not already done)
2. Go to Account Settings → API tokens
3. Create a new token with "Upload" permission
4. Copy the token (format: `pypi-AgEIcHlwaS5vcmc...`)

**Example value:**
```
pypi-AgEIcHlwaS5vcmcAAAAAAAAA-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
```

### 3. `GITHUB_REPO_URL`

**Purpose:** GitHub repository URL for syncing release tags

**Value format:** SSH URL
```
git@github.com:onera/MOLA.git
```

**Note:** The SSH key used by GitLab CI must have write access to the GitHub repository.

### 4. `GITHUB_DEPLOY_KEY`

**Purpose:** SSH private key for pushing tags to GitHub

**How to generate:**
1. Generate a new SSH key pair (if not existing):
   ```bash
   ssh-keygen -t ed25519 -C "gitlab-ci@mola.onera.fr" -f github_deploy_key
   ```
2. Add the **public** key (`github_deploy_key.pub`) to GitHub repository → Settings → Deploy keys → Add deploy key (enable "Allow write access")
3. Copy the **private** key content (`github_deploy_key`)

**Example value:**
```
-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAA...
-----END OPENSSH PRIVATE KEY-----
```

### 5. `MOLA_PASSWORD` (Deprecated)

**Purpose:** Password for the `mola` user account (no longer used)

**Note:** This variable is no longer used. The deployment now uses SSH key-based authentication via `MOLA_DEPLOY_SSH_KEY` (File variable). Password-based deployment was abandoned due to `su` permission restrictions.

### 6. `MOLA_DEPLOY_SSH_KEY_B64` (Deprecated)

**Purpose:** Base64-encoded SSH private key (no longer used)

**Note:** This variable has been deprecated in favor of the File variable approach (`MOLA_DEPLOY_SSH_KEY`). Base64 encoding caused issues with special characters and line breaks. If you have this variable configured, you can safely remove it.

## Tag Protection Configuration

To ensure only maintainers can create release tags:

1. Go to GitLab Project → Settings → Repository → Protected Tags
2. Add protection rules:
   - **Tag pattern:** `v*`
   - **Allowed to create:** Maintainers only

## Pipeline Trigger Workflow

### Release Candidate Pipeline

1. Create and push a release candidate tag:
   ```bash
   git tag v2.0.1-rc1
   git push origin v2.0.1-rc1
   ```
2. Navigate to CI/CD → Pipelines in GitLab
3. The pipeline will appear with manual jobs
4. Click "Play" on `release_candidate_prepare` to start

### Final Release Pipeline

1. After RC validation passes, create final release tag:
   ```bash
   git tag v2.0.1
   git push origin v2.0.1
   ```
2. Navigate to CI/CD → Pipelines in GitLab
3. Click "Play" on `release_final_deploy` to start the full release process

## Troubleshooting

### SSH Key Authentication Fails
- Verify `MOLA_DEPLOY_SSH_KEY` is correctly set as a **File variable** in GitLab CI/CD
- Check that the SSH key permissions are correct on JUNO: `chmod 600 ~/.ssh/id_gitlab_ci`
- Test SSH manually: `ssh -i ~/.ssh/id_gitlab_ci mola@localhost "echo test"`
- Check that `authorized_keys` contains the public key
- Ensure the private key content in GitLab includes the full header and footer lines

### PyPI Upload Fails
- Verify `PYPI_API_TOKEN` is correct and has upload permissions
- Check that the token hasn't expired
- Ensure the package version doesn't already exist on PyPI

### GitHub Sync Fails
- Verify `GITHUB_DEPLOY_KEY` is correctly configured
- Check that the corresponding public key has write access on GitHub
- Ensure `GITHUB_REPO_URL` is correct

### Deployment Fails
- Verify the GitLab runner is running on JUNO
- Check that `MOLA_DEPLOY_SSH_KEY` File variable is correctly configured
- Review the job logs for specific error messages

## Security Notes

- Never commit these variables to the repository
- Rotate SSH keys periodically
- Use separate keys for different environments (dev, staging, production)
- Limit access to CI/CD variable settings to maintainers only
- The `MOLA_DEPLOY_SSH_KEY` variable is masked in logs and protected for release tags only