# -- IMPORTS

import argparse;
import os;
import warnings;
import matplotlib.pyplot as plt;
import numpy as np;
import pandas as pd;
from scipy.optimize import least_squares;

# -- FUNCTIONS

def get_linear_residual_array(
    parameter_array: np.ndarray,
    feature_array: np.ndarray,
    target_array: np.ndarray
    ) -> np.ndarray:

    coefficient_array = parameter_array[ :-1 ];
    intercept = parameter_array[ -1 ];

    predicted_target_array = np.dot( feature_array, coefficient_array ) + intercept;

    return ( predicted_target_array - target_array.flatten() );

# ~~

def generate_svg_plot(
    past_data_frame: pd.DataFrame,
    future_data_frame: pd.DataFrame,
    date_column_name: str,
    price_column_name: str,
    fitted_price_array: np.ndarray,
    output_svg_file_path: str
    ) -> None:

    past_price_array = past_data_frame[ price_column_name ].values;
    future_price_array = future_data_frame[ price_column_name ].values;

    past_count = len( past_price_array );
    future_count = len( future_price_array );
    total_count = past_count + future_count;

    past_index_array = np.arange( past_count );
    future_index_array = np.arange( past_count, total_count );

    all_price_array = np.concatenate( [ past_price_array, future_price_array, fitted_price_array ] );
    price_minimum = all_price_array.min();
    price_maximum = all_price_array.max();

    plt.figure( figsize = ( 20, 12 ) );
    plt.plot( past_index_array, past_price_array, color = '#000000', linewidth = 1, label = 'Past prices' );
    plt.plot( past_index_array, fitted_price_array, color = '#4080FF', linewidth = 1, label = 'Fitted values' );
    plt.plot( future_index_array, future_price_array, color = '#FF0000', linewidth = 1, label = 'Future prices' );

    plt.xlabel( 'Index ( 0 to N )' );
    plt.ylabel( 'Price' );
    plt.title( 'Stock Price Prediction' );
    plt.legend();
    plt.grid( True, alpha = 0.3 );
    plt.ylim( price_minimum, price_maximum );
    plt.xlim( 0, total_count - 1 );

    plt.subplots_adjust( left = 0, bottom = 0, right = 1, top = 1 );
    plt.savefig( output_svg_file_path, format = 'svg', bbox_inches = 'tight', pad_inches = 0 );
    plt.close();

# ~~

def predict_stock_prices(
    input_csv_file_path: str,
    output_folder_path: str,
    date_column_name: str,
    price_column_name: str,
    lookback_count: int = 252,
    prediction_count: int = 14
    ) -> str:

    np.random.seed( 42 );

    stock_data_frame = pd.read_csv( input_csv_file_path );

    if date_column_name not in stock_data_frame.columns or price_column_name not in stock_data_frame.columns:

        raise ValueError( f"Input CSV must contain '{date_column_name}' and '{price_column_name}' columns." );

    stock_data_frame[ date_column_name ] = pd.to_datetime( stock_data_frame[ date_column_name ] );
    stock_data_frame = stock_data_frame.sort_values( by = date_column_name, ascending = True );
    stock_data_frame[ date_column_name ] = stock_data_frame[ date_column_name ].dt.strftime( '%Y-%m-%d' );
    price_array = stock_data_frame[ price_column_name ].astype( float ).values;

    if len( price_array ) < 10:

        raise ValueError( "Time series is too short for meaningful training." );

    lag_count = 3;

    original_price_array = price_array.copy();
    training_start_index = max( 0, len( original_price_array ) - lookback_count );
    training_price_array = original_price_array[ training_start_index: ];

    if len( training_price_array ) < lag_count + 1:

        raise ValueError( f"Time series is too short for {lag_count} lag features." );

    raw_feature_array = [];
    raw_target_array = [];

    for price_index in range( lag_count, len( training_price_array ) - 1 ):

        lag_feature_array = [ training_price_array[ price_index - lag_index ] for lag_index in range( lag_count ) ];
        raw_feature_array.append( lag_feature_array );
        raw_target_array.append( training_price_array[ price_index + 1 ] );

    raw_feature_array = np.array( raw_feature_array );
    raw_target_array = np.array( raw_target_array );

    feature_mean = raw_feature_array.mean( axis = 0 );
    feature_standard_deviation = raw_feature_array.std( axis = 0 ) + 1e-8;
    normalized_feature_array = ( raw_feature_array - feature_mean ) / feature_standard_deviation;

    target_mean = raw_target_array.mean();
    target_standard_deviation = raw_target_array.std() + 1e-8;
    normalized_target_array = ( raw_target_array - target_mean ) / target_standard_deviation;

    initial_parameter_array = np.concatenate( [ np.array( [ 1.0 ] + [ 0.0 ] * ( lag_count - 1 ) ), np.array( [ 0.0 ] ) ] );

    optimization_result_fit = (
        least_squares(
            get_linear_residual_array,
            initial_parameter_array,
            args = ( normalized_feature_array, normalized_target_array ),
            method = "lm",
            max_nfev = 1000,
            ftol = 1e-6
            )
        );

    normalized_coefficients_fit = optimization_result_fit.x[ :-1 ];
    normalized_intercept_fit = optimization_result_fit.x[ -1 ];

    fitted_price_array = np.full( len( original_price_array ), np.nan );
    fitted_price_array[ :lag_count ] = original_price_array[ :lag_count ];

    for price_index in range( lag_count, len( original_price_array ) ):

        lag_feature_array = np.array( [ original_price_array[ price_index - 1 - lag_index ] for lag_index in range( lag_count ) ] );
        normalized_lag_feature_array = ( lag_feature_array - feature_mean ) / feature_standard_deviation;
        normalized_prediction = np.dot( normalized_lag_feature_array, normalized_coefficients_fit ) + normalized_intercept_fit;
        fitted_price = normalized_prediction * target_standard_deviation + target_mean;
        fitted_price_array[ price_index ] = fitted_price;

    prediction_array = [];
    price_array = original_price_array.copy();

    for prediction_index in range( prediction_count ):

        training_end_index = len( price_array );
        training_start_index = max( 0, training_end_index - lookback_count );
        training_price_array = price_array[ training_start_index:training_end_index ];

        if len( training_price_array ) < lag_count + 2:

            raw_feature_array = training_price_array[ :-1 ].reshape( -1, 1 );
            raw_target_array = training_price_array[ 1: ].reshape( -1, 1 );

            feature_mean = raw_feature_array.mean();
            feature_standard_deviation = raw_feature_array.std() + 1e-8;
            normalized_feature_array = ( raw_feature_array - feature_mean ) / feature_standard_deviation;

            target_mean = raw_target_array.mean();
            target_standard_deviation = raw_target_array.std() + 1e-8;
            normalized_target_array = ( raw_target_array - target_mean ) / target_standard_deviation;

            initial_parameter_array = np.array( [ 1.0, 0.0 ] );

            optimization_result = (
                least_squares(
                    get_linear_residual_array,
                    initial_parameter_array,
                    args = ( normalized_feature_array, normalized_target_array ),
                    method = "lm",
                    max_nfev = 1000,
                    ftol = 1e-6
                    )
                );

            normalized_slope = optimization_result.x[ 0 ];
            normalized_intercept = optimization_result.x[ 1 ];

            normalized_prediction_array = normalized_slope * normalized_feature_array.flatten() + normalized_intercept;
            normalized_residual_array = normalized_target_array.flatten() - normalized_prediction_array;
            residual_standard_deviation = normalized_residual_array.std();

            last_known_price = price_array[ -1 ];
            normalized_last_input = ( last_known_price - feature_mean ) / feature_standard_deviation;
            normalized_prediction = normalized_slope * normalized_last_input + normalized_intercept;

            residual_noise = np.random.normal( 0, residual_standard_deviation );
            normalized_prediction = normalized_prediction + residual_noise;
            predicted_price = normalized_prediction * target_standard_deviation + target_mean;

        else:

            raw_feature_array = [];
            raw_target_array = [];

            for price_index in range( lag_count, len( training_price_array ) - 1 ):

                lag_feature_array = [ training_price_array[ price_index - lag_index ] for lag_index in range( lag_count ) ];
                raw_feature_array.append( lag_feature_array );
                raw_target_array.append( training_price_array[ price_index + 1 ] );

            raw_feature_array = np.array( raw_feature_array );
            raw_target_array = np.array( raw_target_array );

            feature_mean = raw_feature_array.mean( axis = 0 );
            feature_standard_deviation = raw_feature_array.std( axis = 0 ) + 1e-8;
            normalized_feature_array = ( raw_feature_array - feature_mean ) / feature_standard_deviation;

            target_mean = raw_target_array.mean();
            target_standard_deviation = raw_target_array.std() + 1e-8;
            normalized_target_array = ( raw_target_array - target_mean ) / target_standard_deviation;

            initial_parameter_array = np.concatenate( [ np.array( [ 1.0 ] + [ 0.0 ] * ( lag_count - 1 ) ), np.array( [ 0.0 ] ) ] );

            optimization_result = (
                least_squares(
                    get_linear_residual_array,
                    initial_parameter_array,
                    args = ( normalized_feature_array, normalized_target_array ),
                    method = "lm",
                    max_nfev = 1000,
                    ftol = 1e-6
                    )
                );

            normalized_coefficients = optimization_result.x[ :-1 ];
            normalized_intercept = optimization_result.x[ -1 ];

            normalized_prediction_array = np.dot( normalized_feature_array, normalized_coefficients ) + normalized_intercept;
            normalized_residual_array = normalized_target_array.flatten() - normalized_prediction_array;
            residual_standard_deviation = normalized_residual_array.std();

            last_lag_feature_array = np.array( [ price_array[ -1 - lag_index ] for lag_index in range( lag_count ) ] );
            normalized_last_input = ( last_lag_feature_array - feature_mean ) / feature_standard_deviation;
            normalized_prediction = np.dot( normalized_last_input, normalized_coefficients ) + normalized_intercept;

            residual_noise = np.random.normal( 0, residual_standard_deviation );
            normalized_prediction = normalized_prediction + residual_noise;
            predicted_price = normalized_prediction * target_standard_deviation + target_mean;

        prediction_array.append( float( predicted_price ) );
        price_array = np.append( price_array, predicted_price );

    last_date_string = stock_data_frame[ date_column_name ].iloc[ -1 ];
    last_date = pd.to_datetime( last_date_string );
    future_date_array = [];

    for date_index in range( prediction_count ):

        future_date = ( last_date + pd.Timedelta( days = date_index + 1 ) ).strftime( '%Y-%m-%d' );
        future_date_array.append( future_date );

    future_data_frame = (
        pd.DataFrame(
            {
                date_column_name: future_date_array,
                price_column_name: prediction_array
            }
            )
        );

    result_data_frame = pd.concat( [ stock_data_frame, future_data_frame ], ignore_index = True );

    base_file_name = os.path.basename( input_csv_file_path );
    file_name, file_extension = os.path.splitext( base_file_name );
    output_csv_file_path = os.path.join( output_folder_path, f"future_{file_name}{file_extension}" );

    os.makedirs( output_folder_path, exist_ok = True );
    result_data_frame.to_csv( output_csv_file_path, index = False );

    output_svg_file_path = os.path.join( output_folder_path, f"future_{file_name}.svg" );
    generate_svg_plot( stock_data_frame, future_data_frame, date_column_name, price_column_name, fitted_price_array, output_svg_file_path );

    print( f"Predictions saved to {output_csv_file_path}" );
    print( f"Chart saved to {output_svg_file_path}" );
    print( f"Last {prediction_count} predicted prices:" );
    print( future_data_frame[ price_column_name ].values );

    return output_csv_file_path;

# ~~

def main():

    parser = (
        argparse.ArgumentParser(
            description = "Usage:\n  python retrofit.py microsoft_data.csv DATE CLOSE 90 FUTURE/\n"
            )
        );
    parser.add_argument( "input_csv_file_path", type = str, help = "Path to input CSV file." );
    parser.add_argument( "date_column_name", type = str, help = "Name of the date column in the CSV." );
    parser.add_argument( "price_column_name", type = str, help = "Name of the price column in the CSV." );
    parser.add_argument( "prediction_count", type = int, help = "Number of future prices to predict." );
    parser.add_argument( "output_folder_path", type = str, help = "Output directory ( will contain *_future.csv )." );

    parsed_argument = parser.parse_args();

    input_csv_file_path = parsed_argument.input_csv_file_path;
    date_column_name = parsed_argument.date_column_name;
    price_column_name = parsed_argument.price_column_name;
    prediction_count = parsed_argument.prediction_count;
    output_folder_path = parsed_argument.output_folder_path;

    if not os.path.isfile( input_csv_file_path ):

        raise FileNotFoundError( f"Input file not found: {input_csv_file_path}" );

    if not os.path.isdir( output_folder_path ):

        raise NotADirectoryError(
            f"Output directory does not exist ( create it first ): {output_folder_path}"
        );

    if prediction_count < 1:

        raise ValueError( "prediction_count must be at least 1" );

    predict_stock_prices( input_csv_file_path, output_folder_path, date_column_name, price_column_name, prediction_count = prediction_count );

# -- STATEMENTS

if __name__ == "__main__":

    main();
