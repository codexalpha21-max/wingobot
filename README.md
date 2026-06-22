# pyapi

## Railway persistent history

Railway's normal container filesystem is replaced on every deployment. Attach a
Railway Volume to this service with mount path `/data`, then set:

```text
PYAPI_DATA_DIR=/data
```

All CSV history, caches, model brains, and training metadata use that directory.
Existing files on the volume are never replaced by files bundled in a new image.
The startup log must show:

```text
[STORAGE] provider=custom persistent=True path=/data
```
