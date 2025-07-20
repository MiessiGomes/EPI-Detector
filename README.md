# PPE Monitor Project – Personal Protective Equipment Detection with Computer Vision

## Overview

This project performs **PPE (Personal Protective Equipment) detection** in images or videos using YOLO (You Only Look Once) models. When a worker is identified **without the mandatory PPE**, an **automatic alert is triggered via webhook to a Discord channel**.

---

## Features

- Detects people and PPE such as: helmets, gloves, boots, masks, earmuffs, goggles, protective clothing.
- Verifies **compliance** between the person and the mandatory PPE.
- Sends **automatic alerts via Discord** if someone is without PPE.
- Modularized with best practices: object orientation, `type hints`, `.env`, logging, and separation of responsibilities.
- Support for processing **images, videos, and live webcam**.

---

## Dataset

The dataset used is in YOLO format with the following classes:

```yaml
names: [
  'Earmuffs', 'Earmuffs with',
  'Safety Boots', 'Safety Helmet',
  'Safety Helmet with', 'Protective Gloves',
  'Protective Gloves with', 'Mask', 'Goggles',
  'Goggles with', 'Person', 'Person with',
  'Protective Clothing'
]
```

---

## Folder Structure

```
epi_monitor/
├── config/
│   └── settings.py          # Loads variables from .env
├── core/
│   ├── detector.py          # YOLO model loading
│   ├── evaluator.py         # Compliance logic
│   └── notifier.py          # Discord alert sending
├── utils/
│   └── helpers.py           # Helper functions (IO, drawing, etc.)
├── main.py                  # Application entry point
├── requirements.txt
├── .env
└── README.md                # Project documentation
```

---

## Environment Variables (.env)

```env
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_WEBHOOK
CONFIDENCE_THRESHOLD=0.5
MODEL_PATH=path/to/best.pt
VIDEO_SOURCE=0  # 0 for webcam, or path to video
```

---

## Detection and Compliance

The verification is based on the following rules:

| Class              | Requirement                                        |
| ------------------ | -------------------------------------------------- |
| Person             | Must be with at least: Helmet, Goggles, Gloves     |
| Person with        | Considered **compliant**                           |
| Person (without "with") | Considered **non-compliant**                     |

In addition, the check compares nearby bounding boxes (e.g., a person and a helmet nearby indicate use).

---

## Discord Alert Webhook

### Alert example:

```
NON-COMPLIANCE ALERT
Person detected without mandatory PPE:
- Helmet
- Goggles
Timestamp: 2025-07-19 14:32:10
```

---

## ▶How to Run the Project

### 1. Install the requirements:

```bash
pip install -r requirements.txt
```

### 2. Configure the `.env` with your webhook and YOLO model path.

### 3. Run the system:

```bash
python main.py
```

---

## Technical Requirements

* Python 3.12
* Ultralytics YOLO (v11)
* OpenCV
* dotenv
* requests (for Discord)

---

## Future Improvements

* Support for multiple simultaneous cameras
* Real-time dashboard with Streamlit
* Violation logging in a database
* Web interface with history and alerts

---

## Author

**José Ricardo**
Specialist in Computer Vision, Deep Learning, and intelligent automation.
GitHub: [github.com/MiessiGomes](https://github.com/josericardocv)
Medium: [medium.com/@miessigomes](https://medium.com/@josericardocv)

---

## License

MIT License
