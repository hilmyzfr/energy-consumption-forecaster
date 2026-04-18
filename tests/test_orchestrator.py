import json
from src.agent.orchestrator import run

email = """
Von: betrieb@metallwerk-hannover.de
Betreff: Betriebsunterbrechung März

Guten Tag,

wir möchten Sie darüber informieren, dass unser Werk in der Zeit vom
10.03.2025 bis 12.03.2025 aufgrund von geplanten Wartungsarbeiten
vollständig abgeschaltet wird. Der Stromverbrauch wird in diesem
Zeitraum gegen null gehen.

Mit freundlichen Grüßen,
Thomas Becker
"""

result = run(
    customer_id="C-1042",
    raw_email=email,
    lag_1=1350.0,
    lag_7=1380.0
)

print(json.dumps(result, indent=2, ensure_ascii=False))
