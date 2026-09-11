FROM node:20-alpine

WORKDIR /app

ENV NODE_ENV=development
ENV HOST=0.0.0.0
ENV PORT=3000
ENV WATCHPACK_POLLING=true
ENV USE_TF=0

COPY frontend/package.json frontend/package-lock.json ./

RUN npm ci --legacy-peer-deps

COPY frontend/ .

EXPOSE 3000

CMD ["npm", "start"]