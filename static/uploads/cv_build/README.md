# CV builder

From the parent `uploads` directory, create the local environment and install the PDF dependency:

```sh
uv venv cv_build/.venv
uv pip install --python cv_build/.venv/bin/python reportlab
```

Edit `cv_config.json`, then rebuild:

```sh
cv_build/.venv/bin/python cv_build/generate_cv.py
```

The default output is `prince_modi_CV_generated.pdf` in the parent directory. To choose a path:

```sh
cv_build/.venv/bin/python cv_build/generate_cv.py --output /path/to/my_cv.pdf
```

The builder reads the config relative to its own location, so it works from any current directory. The source CV PDFs are preserved.
