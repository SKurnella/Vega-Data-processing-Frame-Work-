#ifndef VEGA_VEGADATAFRAME_H
#define VEGA_VEGADATAFRAME_H

#include <stdexcept>
#include <string>
#include <vector>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <unordered_map>
#include <map>
#include <set>
#include <numeric>
#include <random>
#include <cmath>
#include <limits>
#include <regex>
#include <functional>

class FILE_ERROR : public std::runtime_error {
public:
    explicit FILE_ERROR(const std::string & error_message);
};

enum class DataType { INT, FLOAT, STRING };

// Forward declaration
class vegaDataframe;

// Abstract base class for imputation strategies
class Imputer {
public:
    virtual ~Imputer() = default;
    virtual void impute(vegaDataframe& df, const std::string& column) = 0;
};

class vegaDataframe {
public:
    std::vector<std::string> data_features;
    std::vector<std::vector<std::string>> data_values;
    std::vector<size_t> non_null_counts;
    std::vector<DataType> column_types;
    std::vector<std::vector<size_t>> null_positions;

    // ============= CORE DATAFRAME OPERATIONS =============
    //this function reads data from the csv file using input stream of the fle
    void read_csv(const std::string & FILE_NAME);
    //this function reads data from the json file using input stream of the file
    void read_json(const std::string & FILE_NAME);
    //this function displays the information about the file which include
    //size and shape of the data, missing values, possible datatype of the object, null values, data features.
    void info() const;

    void describe() const;
    void head(size_t n = 5) const;
    void tail(size_t n = 5) const;

    // ============= SHAPE AND STRUCTURE =============
    [[nodiscard]] std::pair<size_t, size_t> shape() const;
    [[nodiscard]] std::vector<DataType> dtypes() const;
    [[nodiscard]] std::vector<size_t> isnull() const;
    [[nodiscard]] std::vector<size_t> notnull() const;
    [[nodiscard]] size_t count_nulls() const;
    [[nodiscard]] size_t memory_usage() const;

    // ============= COLUMN OPERATIONS =============
    [[nodiscard]] std::vector<std::string> get_column(const std::string& col_name) const;
    std::vector<std::string> get_column(size_t col_index) const;
    void add_column(const std::string& col_name, const std::vector<std::string>& values);
    void insert_column(size_t pos, const std::string& col_name, const std::vector<std::string>& values);
    void drop_column(const std::string& col_name);
    void drop_columns(const std::vector<std::string>& col_names);
    void rename_column(const std::string& old_name, const std::string& new_name);
    void rename_columns(const std::map<std::string, std::string>& rename_map);
    std::vector<std::string> columns() const;

    // ============= ROW OPERATIONS =============
    vegaDataframe filter_rows(const std::string& col_name, const std::string& value) const;
    vegaDataframe filter_rows(const std::function<bool(const std::vector<std::string>&)>& condition) const;
    vegaDataframe query(const std::string& expression) const;
    void drop_row(size_t row_index);
    void drop_rows(const std::vector<size_t>& row_indices);
    vegaDataframe sample(size_t n, bool replace = false) const;
    vegaDataframe nlargest(size_t n, const std::string& col_name) const;
    vegaDataframe nsmallest(size_t n, const std::string& col_name) const;

    // ============= INDEXING AND SELECTION =============
    vegaDataframe loc(const std::vector<size_t>& rows, const std::vector<std::string>& cols) const;
    vegaDataframe iloc(const std::vector<size_t>& rows, const std::vector<size_t>& cols) const;
    std::string at(size_t row, const std::string& col) const;
    std::string iat(size_t row, size_t col) const;

    // ============= STATISTICAL OPERATIONS =============
    double mean(const std::string& col_name) const;
    double median(const std::string& col_name) const;
    std::string mode(const std::string& col_name) const;
    double std_dev(const std::string& col_name) const;
    double variance(const std::string& col_name) const;
    double min(const std::string& col_name) const;
    double max(const std::string& col_name) const;
    double sum(const std::string& col_name) const;
    double prod(const std::string& col_name) const;
    size_t count(const std::string& col_name) const;
    size_t nunique(const std::string& col_name) const;
    std::map<std::string, size_t> value_counts(const std::string& col_name) const;
    std::vector<double> quantile(const std::string& col_name, const std::vector<double>& q) const;
    std::map<std::string, double> corr() const;
    std::map<std::string, double> cov() const;

    // ============= MISSING DATA HANDLING =============
    vegaDataframe dropna(const std::string& how = "any") const;
    void fillna_with_imputer(const std::string& col_name, Imputer& imputer);
    void fillna_value(const std::string& col_name, const std::string& value);
    void fillna_method(const std::string& col_name, const std::string& method = "ffill");
    vegaDataframe interpolate(const std::string& col_name, const std::string& method = "linear") const;

    // ============= SORTING OPERATIONS =============
    void sort_values(const std::string& col_name, bool ascending = true);
    void sort_values(const std::vector<std::string>& col_names, const std::vector<bool>& ascending);
    void sort_index(bool ascending = true);
    vegaDataframe rank(const std::string& col_name, const std::string& method = "average") const;

    // ============= GROUPING AND AGGREGATION =============
    std::map<std::string, vegaDataframe> groupby(const std::string& col_name) const;
    std::map<std::vector<std::string>, vegaDataframe> groupby(const std::vector<std::string>& col_names) const;
    vegaDataframe aggregate(const std::map<std::string, std::string>& agg_funcs) const;
    vegaDataframe pivot_table(const std::string& values, const std::string& index, const std::string& columns) const;
    vegaDataframe pivot(const std::string& index, const std::string& columns, const std::string& values) const;
    vegaDataframe melt(const std::vector<std::string>& id_vars = {}, const std::vector<std::string>& value_vars = {}) const;

    // ============= DATA TRANSFORMATION =============
    void label_encode(const std::string& col_name);
    vegaDataframe one_hot_encode(const std::string& col_name) const;
    vegaDataframe get_dummies(const std::vector<std::string>& col_names) const;
    void apply_function(const std::string& col_name, const std::function<std::string(const std::string&)>& func);
    vegaDataframe map_values(const std::string& col_name, const std::map<std::string, std::string>& mapping) const;

    // ============= STRING OPERATIONS =============
    vegaDataframe str_contains(const std::string& col_name, const std::string& pattern) const;
    vegaDataframe str_startswith(const std::string& col_name, const std::string& prefix) const;
    vegaDataframe str_endswith(const std::string& col_name, const std::string& suffix) const;
    vegaDataframe str_replace(const std::string& col_name, const std::string& pattern, const std::string& replacement) const;
    vegaDataframe str_upper(const std::string& col_name) const;
    vegaDataframe str_lower(const std::string& col_name) const;
    vegaDataframe str_strip(const std::string& col_name) const;
    std::vector<size_t> str_len(const std::string& col_name) const;

    // ============= MERGING AND JOINING =============
    vegaDataframe merge(const vegaDataframe& other, const std::string& left_col, const std::string& right_col, const std::string& how = "inner") const;
    vegaDataframe merge(const vegaDataframe& other, const std::vector<std::string>& on, const std::string& how = "inner") const;
    static vegaDataframe concat(const std::vector<vegaDataframe>& dataframes, int axis = 0, bool ignore_index = false);
    vegaDataframe join(const vegaDataframe& other, const std::string& how = "left") const;

    // ============= DUPLICATE HANDLING =============
    std::vector<bool> duplicated(const std::vector<std::string>& subset = {}, bool keep_first = true) const;
    vegaDataframe drop_duplicates(const std::vector<std::string>& subset = {}, bool keep_first = true) const;

    // ============= RESHAPING =============
    vegaDataframe transpose() const;
    vegaDataframe stack() const;
    vegaDataframe unstack() const;
    vegaDataframe reindex(const std::vector<size_t>& new_index) const;
    vegaDataframe reset_index(bool drop = false) const;
    vegaDataframe set_index(const std::string& col_name) const;

    // ============= ARITHMETIC OPERATIONS =============
    vegaDataframe add(const vegaDataframe& other) const;
    vegaDataframe subtract(const vegaDataframe& other) const;
    vegaDataframe multiply(const vegaDataframe& other) const;
    vegaDataframe divide(const vegaDataframe& other) const;
    vegaDataframe add_scalar(double value) const;
    vegaDataframe multiply_scalar(double value) const;

    // ============= COMPARISON OPERATIONS =============
    std::vector<std::vector<bool>> eq(const vegaDataframe& other) const;
    std::vector<std::vector<bool>> ne(const vegaDataframe& other) const;
    std::vector<std::vector<bool>> lt(const vegaDataframe& other) const;
    std::vector<std::vector<bool>> le(const vegaDataframe& other) const;
    std::vector<std::vector<bool>> gt(const vegaDataframe& other) const;
    std::vector<std::vector<bool>> ge(const vegaDataframe& other) const;

    // ============= DATETIME OPERATIONS =============
    vegaDataframe to_datetime(const std::string& col_name, const std::string& format = "%Y-%m-%d") const;
    std::vector<int> dt_year(const std::string& col_name) const;
    std::vector<int> dt_month(const std::string& col_name) const;
    std::vector<int> dt_day(const std::string& col_name) const;
    std::vector<int> dt_dayofweek(const std::string& col_name) const;

    // ============= WINDOW FUNCTIONS =============
    std::vector<double> rolling_mean(const std::string& col_name, size_t window) const;
    std::vector<double> rolling_sum(const std::string& col_name, size_t window) const;
    std::vector<double> rolling_std(const std::string& col_name, size_t window) const;
    std::vector<double> expanding_mean(const std::string& col_name) const;
    std::vector<double> cumsum(const std::string& col_name) const;
    std::vector<double> cumprod(const std::string& col_name) const;
    std::vector<double> pct_change(const std::string& col_name, size_t periods = 1) const;

    // ============= EXPORT OPERATIONS =============
    void to_csv(const std::string& filename, bool index = false, char sep = ',') const;
    void to_json(const std::string& filename) const;
    void to_html(const std::string& filename) const;
    void to_excel(const std::string& filename) const;

    // ============= UTILITY OPERATIONS =============
    vegaDataframe copy() const;
    bool empty() const;
    bool equals(const vegaDataframe& other) const;
    std::vector<std::string> unique(const std::string& col_name) const;
    vegaDataframe where(const std::function<bool(const std::vector<std::string>&)>& condition, const std::string& other = "") const;
    vegaDataframe astype(const std::string& col_name, DataType dtype);

    // ============= HELPER METHODS =============
    size_t find_column_index(const std::string& col_name) const;
    void update_stats_after_modification();
    void print_memory_usage() const;
    void validate_dataframe() const;
};

// ============= CONCRETE IMPUTER CLASSES =============
class MeanImputer : public Imputer {
public:
    void impute(vegaDataframe& df, const std::string& column) override;
};

class MedianImputer : public Imputer {
public:
    void impute(vegaDataframe& df, const std::string& column) override;
};

class ModeImputer : public Imputer {
public:
    void impute(vegaDataframe& df, const std::string& column) override;
};

class ConstantImputer : public Imputer {
    std::string fill_value;
public:
    explicit ConstantImputer(const std::string& val) : fill_value(val) {}
    void impute(vegaDataframe& df, const std::string& column) override;
};

class ForwardFillImputer : public Imputer {
public:
    void impute(vegaDataframe& df, const std::string& column) override;
};

class BackwardFillImputer : public Imputer {
public:
    void impute(vegaDataframe& df, const std::string& column) override;
};

class LinearInterpolationImputer : public Imputer {
public:
    void impute(vegaDataframe& df, const std::string& column) override;
};

// ============= UTILITY FUNCTIONS =============
bool is_csv_file_valid(const std::string & file_name);
std::string data_type_to_string(DataType dt);
DataType infer_data_type(const std::string& value);
std::vector<std::string> split_string(const std::string& str, char delimiter);
std::string join_strings(const std::vector<std::string>& strings, const std::string& delimiter);
double safe_stod(const std::string& str, double default_val = 0.0);
bool is_numeric(const std::string& str);
std::string trim_whitespace(const std::string& str);

#endif // VEGA_VEGADATAFRAME_H