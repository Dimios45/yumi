"""Viser decisions for the one-command UMI collection loop."""
import json
import threading

from umi_review_state import ROOT, accept


def add_review_controls(server, capture, decision_path, report, stop_event):
    """Only a successful explicit acceptance can write an accepted manifest."""
    lock = threading.Lock()
    with server.gui.add_folder('Review this demonstration', order=-10):
        status = server.gui.add_markdown(
            f"**{'Checks passed' if report['review_gate_passed'] else 'Checks failed — acceptance disabled'}**\n\n"
            f"Motion within 10 mm / 5°: {report['recorded_tracked_fraction']:.1%}. "
            f"Valid jaw readings: {report['width_valid_fraction']:.1%}.\n\n" +
            ' '.join(report.get('review_reasons', [])) + '\n\n' +
            'Inspect the replay and camera before accepting. Retry retains this attempt as pending.'
        )
        accept_next = server.gui.add_button('Accept & next', disabled=not report['review_gate_passed'])
        accept_finish = server.gui.add_button('Accept & finish', disabled=not report['review_gate_passed'])
        retry = server.gui.add_button('Retry (keep this attempt pending)')
        finish = server.gui.add_button('Finish (keep this attempt pending)')
    buttons = (accept_next, accept_finish, retry, finish)

    def decide(action):
        with lock:
            if stop_event.is_set():
                return
            try:
                manifest = None
                if action.startswith('accept'):
                    manifest = accept(capture, ROOT/'data/umi-runs/accepted')
                result = {'action': action, 'capture': str(capture),
                          'manifest': str(manifest) if manifest else None}
                # Outside the capture directory so decisions do not alter its fingerprint.
                decision_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = decision_path.with_suffix('.tmp')
                temporary.write_text(json.dumps(result, indent=2)+'\n')
                temporary.replace(decision_path)
            except (OSError, ValueError, KeyError) as exc:
                status.content = f'**Could not complete decision:** {exc}'
                return
            for button in buttons:
                button.disabled = True
            print(f"Review decision: {action}" + (f" · {manifest}" if manifest else ''), flush=True)
            stop_event.set()

    accept_next.on_click(lambda _: decide('accept_next'))
    accept_finish.on_click(lambda _: decide('accept_finish'))
    retry.on_click(lambda _: decide('retry'))
    finish.on_click(lambda _: decide('finish'))
