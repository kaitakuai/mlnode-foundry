# Content-Type injector — S4 form of patches/0001-content-type-middleware.
# Needed when Stage 4 builds on a PUBLISHED mlnode image (mode
# "upstream-overlay"), where the Stage-3 patch series never ran. Idempotent:
# a no-op on images built from our own Stage 3, which already carry it.
COPY tools/runner-patches/content-type-injector.py /tmp/content-type-injector.py
RUN python3 /tmp/content-type-injector.py && rm /tmp/content-type-injector.py
