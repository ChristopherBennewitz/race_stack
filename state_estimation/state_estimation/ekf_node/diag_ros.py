"""DiagnosticArray messages from plain dictionaries (the one place the estimators touch diagnostic_msgs)."""
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

LEVELS = {'ok': DiagnosticStatus.OK, 'warn': DiagnosticStatus.WARN, 'error': DiagnosticStatus.ERROR}


def status(name, level, message, values, hardware_id='state_estimation'):
    """One DiagnosticStatus; ``values`` maps key -> number (rendered with 6 significant digits)."""
    msg = DiagnosticStatus()
    msg.level = LEVELS[level]
    msg.name = name
    msg.message = message
    msg.hardware_id = hardware_id
    msg.values = [KeyValue(key=k, value=f'{float(v):.6g}') for k, v in values.items()]
    return msg


def array(stamp, statuses):
    msg = DiagnosticArray()
    msg.header.stamp = stamp
    msg.status = list(statuses)
    return msg
