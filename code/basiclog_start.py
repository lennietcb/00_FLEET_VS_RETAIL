import logging

def main():
    # TODO: All message send to log output
    logging.basicConfig(level=logging.DEBUG,
                    filename="output.log")

    #Log levels:
    logging.debug("This is a debug message")
    logging.info("This is a debug message")
    logging.warning("This is a warning message")
    logging.error("This is an error message")
    logging.critical("This is a critical message")

    # TODO:
if __name__=="__name__":
    main()





