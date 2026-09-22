# GitHub SSH setup on a fresh RunPod Pod

This is a separate, manual bootstrap step -- deliberately **not** part of
`setup_runpod.sh`. It touches private key material, which shouldn't be
generated, copied, or pasted by an automated script.

Run this once per Pod (or once per persistent `/workspace` volume, if keys
are stored there and survive Pod restarts).

## 1. Generate or copy your deploy key(s)

If you use one GitHub identity for everything, a single key is enough and
you can skip the multi-host config below -- just add the key to your
GitHub account or as a repo deploy key and use the default `git@github.com`
remote.

If you use **separate keys per repo/identity** (e.g. one for DynamicDreamer,
one for this dreamerv3 fork), generate a key per identity:

```bash
ssh-keygen -t ed25519 -C "dynamicdreamer-runpod" -f ~/.ssh/id_dynamicdreamer -N ""
ssh-keygen -t ed25519 -C "dreamerv3-runpod"      -f ~/.ssh/id_dreamerv3      -N ""
```

Add each public key (`~/.ssh/id_*.pub`) to the corresponding GitHub
account/repo under Settings -> Deploy keys (or your personal account's SSH
keys).

## 2. Configure per-repo host aliases

Append to `~/.ssh/config` (create it if it doesn't exist, `chmod 600`):

```
Host github-dynamicdreamer
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_dynamicdreamer
    IdentitiesOnly yes

Host github-dreamerv3
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_dreamerv3
    IdentitiesOnly yes
```

```bash
chmod 600 ~/.ssh/config ~/.ssh/id_dynamicdreamer ~/.ssh/id_dreamerv3
```

## 3. Verify

```bash
ssh -T git@github-dynamicdreamer
ssh -T git@github-dreamerv3
```

Each should print `Hi <username>! You've successfully authenticated...`
(GitHub always replies with exit code 1 even on success -- that's normal
for `ssh -T`, only the message matters).

## 4. Clone / set remotes using the alias

```bash
git clone git@github-dreamerv3:Passakorn-Aiamwasu/dreamerv3.git
# or, for an existing clone:
git remote set-url origin git@github-dreamerv3:Passakorn-Aiamwasu/dreamerv3.git
```

## Persisting across Pod restarts

If `/workspace` is a persistent network volume, keep `~/.ssh/` on it (e.g.
symlink `~/.ssh` to `/workspace/.ssh`) so keys and `config` survive a Pod
restart and this whole doc only needs to be followed once per volume, not
once per Pod boot.
