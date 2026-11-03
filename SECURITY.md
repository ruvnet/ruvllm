# Security scope

This repository is a research implementation for a trusted operator. Native filesystem operations support macOS and Linux. The local gateway binds to loopback, requires a generated client key and keeps tool execution with the client.

Do not expose the internal MLX backend to other users or a public network. Treat model generated code as untrusted. The recorded executable code scorer requires macOS Seatbelt and fails closed on unsupported platforms; it is not a general purpose code hosting service.

Native patching rejects existing output paths, symlinks, unapproved changes and mismatched source bytes. The output directory and operator account remain trusted. File and artifact hashes do not establish independent hardware attestation or production authorization.

Never commit runtime credentials, provider account files, SSH keys or private signing material. Public training and evaluation inputs must be reviewed before cloud transfer. Stopping a rented GPU can leave storage charges; destruction and independent absence checks are required.

Report a suspected vulnerability through GitHub's private vulnerability reporting interface when enabled. Otherwise contact the repository owner privately before publishing exploit details. Do not include credentials in an issue.
