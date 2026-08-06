# libnvrtc symlink fix (Hopper, and harmless elsewhere). The engine needs
# libnvrtc.so resolvable as -lnvrtc at /usr/local/lib; the CUDA package ships
# only the versioned libnvrtc.so.12 under the toolkit dir, which the linker
# does not pick up by bare name. Mirrors the entrypoint's existing libcuda.so
# fix ("common on Vast.ai") — Pasha 2026-08-06: without it a Hopper engine
# does not start at all. Idempotent; a no-op if the target already resolves.
RUN set -e; \
    if ! (echo 'int main(){return 0;}' | gcc -x c - -lnvrtc -o /dev/null 2>/dev/null); then \
        real="$(find /usr/local /usr/lib -name 'libnvrtc.so.*' 2>/dev/null | grep -E 'libnvrtc\.so\.[0-9]+$' | head -1)"; \
        if [ -n "$real" ]; then \
            ln -sf "$real" /usr/local/lib/libnvrtc.so && ldconfig; \
            echo "libnvrtc-symlink: linked /usr/local/lib/libnvrtc.so -> $real"; \
        else \
            echo "libnvrtc-symlink: no libnvrtc.so.N found; leaving as-is" >&2; \
        fi; \
    else \
        echo "libnvrtc-symlink: -lnvrtc already resolves; no-op"; \
    fi
