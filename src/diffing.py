import json
import os
import urllib.request
import urllib.error

def generate_html_report(prev_data, curr_data, prev_meta, curr_meta, history_data, regressions, improvements, pass_rate_delta, prev_pass_rate, curr_pass_rate):
    # Prepare historical data for Chart.js
    labels = [h.get("timestamp", "")[:10] + " " + h.get("timestamp", "")[11:16] for h in history_data][-10:]
    scores = [h.get("accuracy", 0) for h in history_data][-10:]
    
    # generate table rows for regressions
    regression_rows = ""
    for r in regressions:
        tc_id = r["id"]
        expected = r["expected"]
        prev_actual = r["prev_actual"]
        curr_actual = r["curr_actual"]
        
        regression_rows += f"""
        <tr>
            <td>{tc_id}</td>
            <td><strong>Category:</strong> {expected.get('category', '')}<br><strong>Summary:</strong> {expected.get('summary', '')}</td>
            <td class="text-success"><strong>Category:</strong> {prev_actual.get('category', '') if prev_actual else 'N/A'}<br><strong>Summary:</strong> {prev_actual.get('summary', '') if prev_actual else 'N/A'}</td>
            <td class="text-danger"><strong>Category:</strong> {curr_actual.get('category', '') if curr_actual else 'N/A'}<br><strong>Summary:</strong> {curr_actual.get('summary', '') if curr_actual else 'N/A'}</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Run Comparison Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .card {{ border: 1px solid #ccc; border-radius: 5px; padding: 15px; margin-bottom: 20px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }}
            h1, h2, h3 {{ color: #333; margin-top: 0; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ccc; padding: 10px; text-align: left; vertical-align: top; }}
            th {{ background-color: #f4f4f4; }}
            .text-success {{ color: green; }}
            .text-danger {{ color: red; }}
            .scorecard {{ display: flex; gap: 20px; }}
            .score-item {{ font-size: 1.2em; }}
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <h1>Eval Run Comparison Report</h1>
        
        <div class="card">
            <h2>Run Metadata</h2>
            <ul>
                <li><strong>Timestamp:</strong> {curr_meta.get('timestamp', 'N/A')}</li>
                <li><strong>Prompt Version:</strong> {curr_meta.get('prompt_version', 'N/A')}</li>
                <li><strong>Model:</strong> {curr_meta.get('model', 'N/A')}</li>
            </ul>
        </div>
        
        <div class="card">
            <h2>Summary Scorecard</h2>
            <div class="scorecard">
                <div class="score-item">
                    <strong>Previous Accuracy:</strong> {prev_pass_rate:.1f}%
                </div>
                <div class="score-item">
                    <strong>Current Accuracy:</strong> {curr_pass_rate:.1f}%
                </div>
                <div class="score-item">
                    <strong>Delta:</strong> <span class="{'text-success' if pass_rate_delta >= 0 else 'text-danger'}">{pass_rate_delta:+.1f}%</span>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Trend Chart (Last {len(scores)} Runs)</h2>
            <canvas id="trendChart" width="400" height="100"></canvas>
            <script>
                var ctx = document.getElementById('trendChart').getContext('2d');
                var trendChart = new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        labels: {json.dumps(labels)},
                        datasets: [{{
                            label: 'Accuracy (%)',
                            data: {json.dumps(scores)},
                            borderColor: 'blue',
                            fill: false,
                            tension: 0.1
                        }}]
                    }},
                    options: {{
                        scales: {{
                            y: {{ beginAtZero: true, max: 100 }}
                        }}
                    }}
                }});
            </script>
        </div>
        
        <div class="card">
            <h2>Regressions ({len(regressions)})</h2>
            {f'''
            <table>
                <tr>
                    <th>Test ID</th>
                    <th>Expected</th>
                    <th>Old Output (Passed)</th>
                    <th>New Output (Failed)</th>
                </tr>
                {regression_rows}
            </table>
            ''' if regressions else '<p>No regressions found!</p>'}
        </div>
        
    </body>
    </html>
    """
    
    with open("diff_report.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("HTML report generated at diff_report.html")

def send_slack_alert(pass_rate_delta, prev_pass_rate, curr_pass_rate, num_regressions, report_url=None):
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print("SLACK_WEBHOOK_URL not set. Skipping Slack alert.")
        return
        
    warning_threshold = 3.0
    critical_threshold = 8.0
    
    if pass_rate_delta <= -critical_threshold:
        status = "FAIL"
        color = "#ff0000"
        emoji = "❌"
    elif pass_rate_delta <= -warning_threshold:
        status = "WARN"
        color = "#ffcc00"
        emoji = "⚠️"
    else:
        status = "PASS"
        color = "#36a64f"
        emoji = "✅"
    
    direction = "dropped" if pass_rate_delta < 0 else "increased"
    
    if pass_rate_delta == 0:
        headline = f"Accuracy remained stable at {curr_pass_rate:.1f}%."
    else:
        headline = f"Accuracy {direction} from {prev_pass_rate:.1f}% to {curr_pass_rate:.1f}%."
        
    if num_regressions > 0:
        headline = f"{num_regressions} regressions detected, " + headline
    
    text_msg = f"{emoji} *Eval Run Status: {status}*\n{headline}"
    if report_url:
        text_msg += f"\n<{report_url}|View full HTML diff report>"
        
    payload = {
        "text": text_msg,
        "attachments": [
            {
                "color": color,
                "fields": [
                    {
                        "title": "Previous Accuracy",
                        "value": f"{prev_pass_rate:.1f}%",
                        "short": True
                    },
                    {
                        "title": "Current Accuracy",
                        "value": f"{curr_pass_rate:.1f}%",
                        "short": True
                    },
                    {
                        "title": "Regressions",
                        "value": str(num_regressions),
                        "short": True
                    }
                ]
            }
        ]
    }
    
    req = urllib.request.Request(
        webhook_url, 
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            print(f"Slack alert sent. Status code: {response.getcode()}")
    except urllib.error.URLError as e:
        print(f"Failed to send Slack alert: {e}")

def compare_runs(prev_file: str, curr_file: str):
    with open(prev_file, "r") as f:
        prev_run = json.load(f)
    with open(curr_file, "r") as f:
        curr_run = json.load(f)
        
    prev_data = prev_run.get("results", prev_run) if isinstance(prev_run, dict) else prev_run
    curr_data = curr_run.get("results", curr_run) if isinstance(curr_run, dict) else curr_run
    
    prev_meta = prev_run.get("metadata", {}) if isinstance(prev_run, dict) else {}
    curr_meta = curr_run.get("metadata", {}) if isinstance(curr_run, dict) else {}

    prev_map = {item["id"]: item for item in prev_data}

    regressions = []
    improvements = []
    prev_passes = 0
    curr_passes = 0
    categories = {}

    for curr in curr_data:
        tc_id = curr["id"]
        if tc_id not in prev_map:
            continue
            
        prev = prev_map[tc_id]
        
        prev_passed = prev.get("category_score", 0) == 1
        curr_passed = curr.get("category_score", 0) == 1
        
        if prev_passed: prev_passes += 1
        if curr_passed: curr_passes += 1
        
        if prev_passed and not curr_passed:
            # regression
            prev_actual = prev.get("actual", {}).get("result", {}) if prev.get("actual") else None
            curr_actual = curr.get("actual", {}).get("result", {}) if curr.get("actual") else None
            expected = {"category": curr.get("expected_category", ""), "summary": curr.get("expected_summary", "")}
            regressions.append({
                "id": tc_id,
                "expected": expected,
                "prev_actual": prev_actual,
                "curr_actual": curr_actual
            })
        elif not prev_passed and curr_passed:
            improvements.append(tc_id)
            
        cat = curr.get("expected_category", "unknown")
        if cat not in categories:
            categories[cat] = {"prev_correct": 0, "curr_correct": 0, "total": 0}
            
        categories[cat]["total"] += 1
        if prev_passed: categories[cat]["prev_correct"] += 1
        if curr_passed: categories[cat]["curr_correct"] += 1

    total_cases = len(curr_data)
    prev_pass_rate = (prev_passes / total_cases) * 100 if total_cases > 0 else 0
    curr_pass_rate = (curr_passes / total_cases) * 100 if total_cases > 0 else 0
    pass_rate_delta = curr_pass_rate - prev_pass_rate
    
    print("\n=== Run Comparison Report ===")
    print(f"Overall accuracy : {prev_pass_rate:.1f}% -> {curr_pass_rate:.1f}%  ({pass_rate_delta:+.1f}%)")
    
    if regressions:
        print(f"\n⚠  Regressions  ({len(regressions)}): {', '.join([r['id'] for r in regressions])}")
    if improvements:
        print(f"Improvements ({len(improvements)}): {', '.join(improvements)}")
        
    history_data = []
    if os.path.exists("run_history.json"):
        try:
            with open("run_history.json", "r") as f:
                history_data = json.load(f)
        except Exception:
            pass
            
    generate_html_report(
        prev_data, curr_data, prev_meta, curr_meta, history_data, 
        regressions, improvements, pass_rate_delta, prev_pass_rate, curr_pass_rate
    )
    
    report_url = os.environ.get("REPORT_URL", f"file://{os.path.abspath('diff_report.html')}")
    send_slack_alert(pass_rate_delta, prev_pass_rate, curr_pass_rate, len(regressions), report_url)

if __name__ == "__main__":
    compare_runs("run_v1.json", "run_v2.json")