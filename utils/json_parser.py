import json
import numpy
import csv

__author__ = 'Simon & Oskar'

def parse_json():
    fields = ['requester_account_age_in_days_at_request',
              'requester_days_since_first_post_on_raop_at_request',
              'requester_number_of_posts_on_raop_at_request',
              'requester_number_of_posts_at_request',
              'requester_number_of_comments_at_request',
              'requester_number_of_comments_in_raop_at_request',
              'requester_number_of_subreddits_at_request',
              'requester_upvotes_minus_downvotes_at_request',
              'unix_timestamp_of_request',
              'requester_received_pizza']

    features = ['acc_age',
                'days_since_first_roap_post',
                'n_posts_on_roap',
                'n_posts_total',
                'n_comments_total',
                'n_comments_in_roap',
                'n_subreddits',
                'upvotes_minus_downvotes',
                'time_stamp',
                'received_pizza']

    json_file = open('resources/train.json', 'r')
    json_data = json.loads(json_file.read())
    json_file.close()

    out_file = open('resources/raop.csv', 'w+')
    out_file.write('{}\n'.format(','.join(features)))

    for entry in json_data:
        row = ','.join([str(float(entry[field])) for field in fields])
        out_file.write('{}\n'.format(row))