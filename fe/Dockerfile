FROM node:20-bookworm-slim AS build

WORKDIR /workspace/fe

COPY package*.json ./
RUN npm ci

COPY . .

ARG VITE_GOOGLE_CLIENT_ID=""
ENV VITE_API_URL=""
ENV VITE_GOOGLE_CLIENT_ID=${VITE_GOOGLE_CLIENT_ID}

RUN npm run build

FROM caddy:2.8-alpine

WORKDIR /usr/share/caddy

COPY Caddyfile /etc/caddy/Caddyfile
COPY --from=build /workspace/fe/dist /usr/share/caddy

EXPOSE 80

CMD ["caddy", "run", "--config", "/etc/caddy/Caddyfile", "--adapter", "caddyfile"]
