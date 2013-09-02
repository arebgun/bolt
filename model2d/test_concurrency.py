#!/usr/bin/env python

from automain import automain 
from argparse import ArgumentParser
from random import choice
import string

from test_model import User, get_session_factory

from multiprocessing import Pool

def create_user(kwargs):
	session = session_factory()
	print kwargs
	new_user = User(**kwargs)
	session.add(new_user)
	session.commit()
	q = session.query(User)
	print q.count()

def random_user():
	N_first = choice(range(3,10))
	N_last = choice(range(3,10))
	N_pass = choice(range(10,20))

	first_name = ''.join([choice(string.ascii_uppercase)]+
		[choice(string.ascii_lowercase) for _ in range(N_first)])
	last_name = ''.join([choice(string.ascii_uppercase)]+
		[choice(string.ascii_lowercase) for _ in range(N_last)])
	password = ''.join(
		choice(string.ascii_lowercase + string.digits) 
		for x in range(N_pass))

	return dict(name=first_name,
				fullname=first_name + ' ' + last_name,
				password=password)

@automain
def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--db_url', required=True, type=str)
    args = parser.parse_args()

    global session_factory
    session_factory = get_session_factory(args.db_url, echo=False)

    num_processors = 7
    p = Pool(num_processors)

    N = 100
    args_list = [random_user() for _ in range(N)]

    p.map(create_user, args_list)