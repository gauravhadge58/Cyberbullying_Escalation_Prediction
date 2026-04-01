FROM node:18-alpine

WORKDIR /app

# Copy dependency configs
COPY package*.json ./
RUN npm install

# Rest of application mapped via volumes
COPY . .

EXPOSE 5000

# Runs via nodemon if in dependencies, or default npm start
CMD ["npm", "run", "dev"]
