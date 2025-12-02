# L.O.V.E application volume
# This contains just the application code

FROM i386/debian:bookworm AS builder
COPY . /app
# Ensure everything is writable by everyone (simplest fix for single-user WebVM)
RUN chmod -R 777 /app

FROM scratch
COPY --from=builder /app /
