# Rasa + Saleor GraphQL (Minimal)

A minimal setup for integrating **[Rasa](https://rasa.com/)** with **[Saleor GraphQL](https://docs.saleor.io/docs/3.x/developer/api/overview/)**.
This repository helps you train a conversational agent and query Saleor for product information.

---

## 🚀 Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # On Linux / macOS
.venv\Scripts\activate      # On Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or, if using [uv](https://github.com/astral-sh/uv):

```bash
uv pip install -r requirements.txt
```

### 3. Configure environment

Copy the example environment file and fill in your Saleor GraphQL credentials:

```bash
cp .env.example .env
```

### 4. Set RASA Pro License

You’ll need to set the `RASA_PRO_LICENSE` environment variable before running Rasa.

* **Linux / macOS (Bash/Zsh):**

  ```bash
  export RASA_PRO_LICENSE="eyJhb..."
  ```

* **Windows (PowerShell):**

  ```powershell
  $env:RASA_PRO_LICENSE="eyJhb..."
  ```

---

## 🏋️ Train and Run

### 1. Train the model

```bash
rasa train -d domain -c configs/config.yml
```

### 2. Run services

Open two terminals:

**Terminal 1** – Run custom actions

```bash
rasa run actions
```

**Terminal 2** – Run Rasa server with API enabled

```bash
rasa run --enable-api -p 5005
```

---

## 💬 Test the Assistant

### Option A: Using REST API

**Linux / macOS (cURL):**

```bash
curl -s localhost:5005/webhooks/rest/webhook \
  -H "Content-Type: application/json" \
  -d '{"sender":"test","message":"What is the price of juice"}'
```

**Windows (PowerShell):**

```powershell
Invoke-RestMethod -Uri "http://localhost:5005/webhooks/rest/webhook" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"sender":"test","message":"What is the price of juice"}'
```

### Option B: Using `rasa inspect`

You can run Rasa in **inspect mode** (no need to start the action server separately):

```bash
rasa inspect --debug
```

---

## 📌 Notes

* Make sure your `.env` file contains valid Saleor credentials.
* Ensure `RASA_PRO_LICENSE` is set before running training or serving.
* You can change the Rasa configuration inside `configs/config.yml`.
* By default, the Rasa server runs on port **5005**.