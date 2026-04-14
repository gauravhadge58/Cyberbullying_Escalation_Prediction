FROM node:18-alpine

WORKDIR /app

# Copy dependency configs
COPY package*.json ./
RUN npm install --omit=dev

COPY . .

EXPOSE 5000

# Production: use node directly, not nodemon
CMD ["npm", "start"]
