import json
import pandas
import numpy
import os

__author__ = 'Simon & Oskar'


def parse_json(json_path, csv_save_path):
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

    script_path = os.path.dirname(__file__)
    json_file_path = os.path.join(script_path, json_path)
    json_file = open(json_file_path, 'r')
    json_data = json.loads(json_file.read())
    json_file.close()

    script_path = os.path.dirname(__file__)
    csv_file_path = os.path.join(script_path, csv_save_path)
    out_file = open(csv_file_path, 'w+')
    out_file.write('{}\n'.format(','.join(features)))

    for entry in json_data:
        row = ','.join([str(float(entry[field])) for field in fields])
        out_file.write('{}\n'.format(row))


def parse_csv(relative_file_path):
    script_path = os.path.dirname(__file__)
    file_path = os.path.join(script_path, relative_file_path)
    csv_data = pandas.read_csv(file_path)

    csv_data = csv_data.reindex(numpy.random.permutation(csv_data.index))

    class_set = csv_data.as_matrix(columns = csv_data.columns[-1:])
    class_set = numpy.array(class_set)
    try:
        class_set = class_set.astype(numpy.float)
    except:
        pass

    feature_set = csv_data.as_matrix(columns = csv_data.columns[:-1])
    feature_set = numpy.array(feature_set).astype('float')
    # feature_set = numpy.around(feature_set, decimals = 8)

    return class_set, feature_set
