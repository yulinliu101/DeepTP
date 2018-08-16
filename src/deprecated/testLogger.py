import logging
import os
def test(x):
    for k in range(x):
        print(k)
        if k % 10 == 0:
            logging.info('yyyyy')
            logger.info('???')
            logger.debug('???')

# to run in console
if __name__ == '__main__':
    import click
    # Use click to parse command line arguments
    @click.command()



    # Train RNN model using a given configuration file
    def main():
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                            filename='myapp.log',
                            filemode='w')
        global logger
        logger = logging.getLogger(os.path.basename(__file__))
        
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
        logger.addHandler(consoleHandler)
        test(50)
    main()
