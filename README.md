# AI Visibility Score System

A web application to generate comprehensive AI visibility reports for any brand using OpenAI's API.

## Features

- Simple and minimal web interface
- Auto-generates brand-specific question sets
- Calculates visibility scores across 4 pillars:
  - Presence (do we show up?)
  - Prominence (how early/strongly are we mentioned?)
  - Narrative Quality (do we sound correct + on-message?)
  - Source Authority (does AI reference our official pages?)
- Generates detailed markdown reports

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Fill in the form with:
   - **Brand Name**: The name of your brand
   - **Industry**: The industry your brand operates in
   - **Brand Description**: A detailed description of your brand, products, and key features
   - **OpenAI API Key**: Your OpenAI API key (starts with "sk-")

4. Click "Generate Visibility Report" and wait for the analysis to complete

## How It Works

1. The system uses AI to generate a custom question set based on your brand and industry
2. It queries the AI model with these questions to see how your brand is represented
3. It analyzes the responses across multiple dimensions
4. It generates a comprehensive report with scores and recommendations

## Output

The report includes:
- Overall visibility score (0-100)
- Component scores for each pillar
- Auto-generated brand configuration
- Question set used
- Per-question analysis
- Recommendations for improvement

## Notes

- The report generation may take a few minutes depending on the number of questions
- Make sure you have a valid OpenAI API key with sufficient credits
- The report can be copied to clipboard using the "Copy Report" button
