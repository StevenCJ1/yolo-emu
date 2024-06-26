import logging

# Logging Setting

log_cmd = logging.getLogger("console-logger")
stream_handler = logging.StreamHandler()
log_cmd.addHandler(stream_handler)
log_cmd.setLevel(logging.INFO)

log_file = logging.getLogger("file-logger")
file_handler = logging.FileHandler("./test.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s %(filename)s %(funcName)s[line:%(lineno)d]%(levelname)s - %(message)s"))
log_file.addHandler(file_handler)
log_file.setLevel(logging.DEBUG)


log_file.debug('log_file.debug')
log_file.info('log_file-info')
log_cmd.info('log_cmd-info')
log_cmd.debug('log_cmd-debug')