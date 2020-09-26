import unittest


# initialize the test suite
loader = unittest.TestLoader()
start_dir = './test/'
suite = loader.discover(start_dir)

# initialize a runner, pass it your suite and run it
runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)