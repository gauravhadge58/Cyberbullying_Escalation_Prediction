FROM node:18-alpine

WORKDIR /app

# Copy dependency files first to utilize Docker build cache
COPY package*.json ./
RUN npm install

# The rest of the app files will be mapped via volumes for hot reloading
# But we copy them in case the image is run standalone
COPY . .

EXPOSE 5173

# Ensures Vite binds to all network interfaces inside the container
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
