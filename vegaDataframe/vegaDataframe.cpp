#include "vegaDataframe.h"
#include <fstream>
#include <sstream>
#include <chrono>
#include <ranges>
// #include <ctime>

// ============= UTILITY FUNCTIONS =============

FILE_ERROR::FILE_ERROR(const std::string & error_message)
    : std::runtime_error(error_message) {}

bool is_csv_file_valid(const std::string & file_name) {
    std::filesystem::path file_path = std::filesystem::current_path() / file_name;

    if (!std::filesystem::exists(file_path) || !std::filesystem::is_regular_file(file_path)) {
        throw FILE_ERROR("File does not exist or it is not a regular file: " + file_name);
    }
    if (file_path.extension() != ".csv") {
        throw FILE_ERROR("Provided file is not a CSV file: " + file_name);
    }
    return true;
}

std::string data_type_to_string(DataType dt) {
    switch (dt) {
        case DataType::INT: return "int";
        case DataType::FLOAT: return "float";
        case DataType::STRING: return "string";
        default: return "unknown";
    }
}

DataType infer_data_type(const std::string& value) {
    if (value.empty()) return DataType::STRING;

    std::istringstream iss(value);
    long long tmp_int;
    if ((iss >> tmp_int) && iss.eof()) {
        return DataType::INT;
    }

    iss.clear();
    iss.str(value);
    double tmp_double;
    if ((iss >> tmp_double) && iss.eof()) {
        return DataType::FLOAT;
    }

    return DataType::STRING;
}

std::vector<std::string> split_string(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

std::string join_strings(const std::vector<std::string>& strings, const std::string& delimiter) {
    if (strings.empty()) return "";

    std::string result = strings[0];
    for (size_t i = 1; i < strings.size(); ++i) {
        result += delimiter + strings[i];
    }
    return result;
}

double safe_stod(const std::string& str, double default_val) {
    try {
        return std::stod(str);
    } catch (...) {
        return default_val;
    }
}

bool is_numeric(const std::string& str) {
    if (str.empty()) return false;
    std::istringstream iss(str);
    double d;
    return (iss >> d) && iss.eof();
}

std::string trim_whitespace(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = str.find_last_not_of(" \t\r\n");
    return str.substr(start, end - start + 1);
}

// ============= VEGADATAFRAME HELPER METHODS =============

size_t vegaDataframe::find_column_index(const std::string& col_name) const {
    auto it = std::ranges::find(data_features, col_name);
    if (it == data_features.end())
        throw std::runtime_error("Column not found: " + col_name);
    return std::distance(data_features.begin(), it);
}

void vegaDataframe::update_stats_after_modification() {
    size_t column_count = data_features.size();
    non_null_counts.assign(column_count, 0);
    null_positions.assign(column_count, std::vector<size_t>{});

    for (size_t row = 0; row < data_values.size(); ++row) {
        for (size_t col = 0; col < column_count && col < data_values[row].size(); ++col) {
            if (data_values[row][col].empty()) {
                null_positions[col].push_back(row);
            } else {
                non_null_counts[col]++;
            }
        }
    }
}

void vegaDataframe::print_memory_usage() const {
    size_t total_memory = 0;

    // Calculate memory for data_values
    for (const auto& row : data_values) {
        for (const auto& cell : row) {
            total_memory += cell.size();
        }
    }

    // Add metadata memory
    total_memory += data_features.size() * sizeof(std::string);
    total_memory += non_null_counts.size() * sizeof(size_t);
    total_memory += column_types.size() * sizeof(DataType);

    std::cout << "Memory usage: " << total_memory << " bytes (" << total_memory / 1024.0 << " KB)" << std::endl;
}

void vegaDataframe::validate_dataframe() const {
    if (data_features.size() != column_types.size()) {
        throw std::runtime_error("DataFrame validation failed: features and types size mismatch");
    }

    for (const auto& row : data_values) {
        if (row.size() > data_features.size()) {
            throw std::runtime_error("DataFrame validation failed: row has more columns than features");
        }
    }
}

// ============= CORE DATAFRAME OPERATIONS =============

void vegaDataframe::read_csv(const std::string & FILE_NAME) {
    is_csv_file_valid(FILE_NAME);

    std::ifstream input_csv_file(FILE_NAME);
    if (!input_csv_file) throw FILE_ERROR("Cannot open file: " + FILE_NAME);

    data_features.clear();
    data_values.clear();
    non_null_counts.clear();
    column_types.clear();
    null_positions.clear();

    std::string header_line;
    if (std::getline(input_csv_file, header_line)) {
        auto tokens = split_string(header_line, ',');
        for (auto& token : tokens) {
            token = trim_whitespace(token);
            if (!token.empty()) {
                data_features.push_back(token);
            }
        }
    }

    size_t column_count = data_features.size();
    non_null_counts.resize(column_count, 0);
    column_types.assign(column_count, DataType::INT);
    null_positions.resize(column_count);

    size_t row_index = 0;
    std::string data_line;

    while (std::getline(input_csv_file, data_line)) {
        auto row_tokens = split_string(data_line, ',');
        std::vector<std::string> data_row;

        for (size_t i = 0; i < column_count; ++i) {
            std::string cell = (i < row_tokens.size()) ? trim_whitespace(row_tokens[i]) : "";
            data_row.push_back(cell);

            if (cell.empty()) {
                null_positions[i].push_back(row_index);
            } else {
                non_null_counts[i]++;
                DataType inferred_type = infer_data_type(cell);
                if (inferred_type > column_types[i]) {
                    column_types[i] = inferred_type;
                }
            }
        }

        data_values.push_back(std::move(data_row));
        row_index++;
    }
}

void vegaDataframe::read_json(const std::string & FILE_NAME) {
    std::ifstream file(FILE_NAME);
    if (!file) throw FILE_ERROR("Cannot open JSON file: " + FILE_NAME);

    // Simple JSON parsing for array of objects
    std::string line;
    bool first_object = true;

    while (std::getline(file, line)) {
        line = trim_whitespace(line);
        if (line.empty() || line == "[" || line == "]") continue;

        if (line.back() == ',') line.pop_back();

        // Parse JSON object (simplified)
        if (line.front() == '{' && line.back() == '}') {
            line = line.substr(1, line.length() - 2);
            auto pairs = split_string(line, ',');

            std::vector<std::string> row;
            for (const auto& pair : pairs) {
                auto kv = split_string(pair, ':');
                if (kv.size() == 2) {
                    std::string key = trim_whitespace(kv[0]);
                    std::string value = trim_whitespace(kv[1]);

                    // Remove quotes
                    if (key.front() == '"' && key.back() == '"') {
                        key = key.substr(1, key.length() - 2);
                    }
                    if (value.front() == '"' && value.back() == '"') {
                        value = value.substr(1, value.length() - 2);
                    }

                    if (first_object) {
                        data_features.push_back(key);
                    }
                    row.push_back(value);
                }
            }

            if (first_object) {
                column_types.assign(data_features.size(), DataType::STRING);
                non_null_counts.resize(data_features.size(), 0);
                null_positions.resize(data_features.size());
                first_object = false;
            }

            data_values.push_back(row);
        }
    }

    update_stats_after_modification();
}

void vegaDataframe::info() const {
    std::cout << "<class 'vegaDataframe'>\n";
    std::cout << "RangeIndex: " << data_values.size()
              << " entries, 0 to " << (data_values.empty() ? 0 : data_values.size() - 1) << "\n";
    std::cout << "Data columns (total " << data_features.size() << " columns):\n";
    std::cout << " #   Column           Non-Null Count  Dtype     Null Count\n";

    for (size_t col = 0; col < data_features.size(); ++col) {
        size_t null_count = null_positions[col].size();

        std::cout << std::setw(2) << col << "  "
                  << std::setw(15) << data_features[col] << "   "
                  << std::setw(13) << non_null_counts[col] << "   "
                  << std::setw(8) << data_type_to_string(column_types[col]) << "   "
                  << std::setw(10) << null_count << "\n";
    }

    size_t int_count = 0, float_count = 0, string_count = 0;
    for (DataType dt : column_types) {
        if (dt == DataType::INT) int_count++;
        else if (dt == DataType::FLOAT) float_count++;
        else if (dt == DataType::STRING) string_count++;
    }
    std::cout << "dtypes: int(" << int_count << "), float(" << float_count << "), string(" << string_count << ")\n";
}

void vegaDataframe::describe() const {
    std::cout << "Statistical Summary:\n";
    std::cout << std::setw(15) << "Column" << std::setw(10) << "Count" << std::setw(10) << "Mean"
              << std::setw(10) << "Std" << std::setw(10) << "Min" << std::setw(10) << "25%"
              << std::setw(10) << "50%" << std::setw(10) << "75%" << std::setw(10) << "Max" << "\n";

    for (size_t i = 0; i < data_features.size(); ++i) {
        if (column_types[i] != DataType::STRING) {
            const std::string& col_name = data_features[i];
            try {
                double mean_val = mean(col_name);
                double std_val = std_dev(col_name);
                double min_val = min(col_name);
                double max_val = max(col_name);

                std::vector<double> q = {0.25, 0.5, 0.75};
                auto quantiles = quantile(col_name, q);

                std::cout << std::setw(15) << col_name
                          << std::setw(10) << non_null_counts[i]
                          << std::setw(10) << std::fixed << std::setprecision(2) << mean_val
                          << std::setw(10) << std::fixed << std::setprecision(2) << std_val
                          << std::setw(10) << std::fixed << std::setprecision(2) << min_val
                          << std::setw(10) << std::fixed << std::setprecision(2) << quantiles[0]
                          << std::setw(10) << std::fixed << std::setprecision(2) << quantiles[1]
                          << std::setw(10) << std::fixed << std::setprecision(2) << quantiles[2]
                          << std::setw(10) << std::fixed << std::setprecision(2) << max_val << "\n";
            } catch (...) {
                std::cout << std::setw(15) << col_name << "   (error computing stats)\n";
            }
        }
    }
}

void vegaDataframe::head(size_t n) const {
    size_t row_count = std::min(n, data_values.size());

    if (row_count == 0) {
        std::cout << "No data rows to display.\n";
        return;
    }

    for (const auto& col_name : data_features) {
        std::cout << std::setw(15) << col_name << " ";
    }
    std::cout << "\n";

    for (size_t i = 0; i < row_count; ++i) {
        const auto& row = data_values[i];
        for (size_t j = 0; j < data_features.size(); ++j) {
            if (j < row.size())
                std::cout << std::setw(15) << row[j] << " ";
            else
                std::cout << std::setw(15) << "" << " ";
        }
        std::cout << "\n";
    }
}

void vegaDataframe::tail(size_t n) const {
    size_t row_count = std::min(n, data_values.size());
    if (row_count == 0) {
        std::cout << "No data rows to display\n";
        return;
    }

    for (const auto& col_name : data_features) {
        std::cout << std::setw(15) << col_name << " ";
    }
    std::cout << "\n";

    size_t start_idx = data_values.size() - row_count;
    for (size_t i = start_idx; i < data_values.size(); ++i) {
        const auto& row = data_values[i];
        for (size_t j = 0; j < data_features.size(); ++j) {
            if (j < row.size())
                std::cout << std::setw(15) << row[j] << " ";
            else
                std::cout << std::setw(15) << "" << " ";
        }
        std::cout << "\n";
    }
}

// ============= SHAPE AND STRUCTURE OPERATIONS =============

std::pair<size_t, size_t> vegaDataframe::shape() const {
    return {data_values.size(), data_features.size()};
}

std::vector<DataType> vegaDataframe::dtypes() const {
    return column_types;
}

std::vector<size_t> vegaDataframe::isnull() const {
    std::vector<size_t> null_counts;
    for (const auto& positions : null_positions) {
        null_counts.push_back(positions.size());
    }
    return null_counts;
}

std::vector<size_t> vegaDataframe::notnull() const {
    return non_null_counts;
}

size_t vegaDataframe::count_nulls() const {
    size_t total = 0;
    for (const auto& positions : null_positions) {
        total += positions.size();
    }
    return total;
}

size_t vegaDataframe::memory_usage() const {
    size_t total_memory = 0;

    for (const auto& row : data_values) {
        for (const auto& cell : row) {
            total_memory += cell.capacity();
        }
    }

    total_memory += data_features.size() * sizeof(std::string);
    total_memory += non_null_counts.size() * sizeof(size_t);
    total_memory += column_types.size() * sizeof(DataType);

    return total_memory;
}

// ============= COLUMN OPERATIONS =============

std::vector<std::string> vegaDataframe::get_column(const std::string& col_name) const {
    size_t col_idx = find_column_index(col_name);
    return get_column(col_idx);
}

std::vector<std::string> vegaDataframe::get_column(size_t col_index) const {
    if (col_index >= data_features.size()) {
        throw std::runtime_error("Column index out of range");
    }

    std::vector<std::string> column;
    for (const auto& row : data_values) {
        if (col_index < row.size()) {
            column.push_back(row[col_index]);
        } else {
            column.push_back("");
        }
    }
    return column;
}

void vegaDataframe::add_column(const std::string& col_name, const std::vector<std::string>& values) {
    if (values.size() != data_values.size())
        throw std::runtime_error("Column size does not match number of rows");

    data_features.push_back(col_name);
    column_types.push_back(DataType::STRING);
    non_null_counts.push_back(0);
    null_positions.push_back(std::vector<size_t>{});

    size_t col_idx = data_features.size() - 1;
    for (size_t i = 0; i < data_values.size(); ++i) {
        data_values[i].push_back(values[i]);
        if (values[i].empty()) {
            null_positions[col_idx].push_back(i);
        } else {
            non_null_counts[col_idx]++;
        }
    }
}

void vegaDataframe::insert_column(size_t pos, const std::string& col_name, const std::vector<std::string>& values) {
    if (pos > data_features.size()) {
        throw std::runtime_error("Insert position out of range");
    }
    if (values.size() != data_values.size()) {
        throw std::runtime_error("Column size does not match number of rows");
    }

    data_features.insert(data_features.begin() + pos, col_name);
    column_types.insert(column_types.begin() + pos, DataType::STRING);
    non_null_counts.insert(non_null_counts.begin() + pos, 0);
    null_positions.insert(null_positions.begin() + pos, std::vector<size_t>{});

    for (size_t i = 0; i < data_values.size(); ++i) {
        data_values[i].insert(data_values[i].begin() + pos, values[i]);
        if (values[i].empty()) {
            null_positions[pos].push_back(i);
        } else {
            non_null_counts[pos]++;
        }
    }
}

void vegaDataframe::drop_column(const std::string& col_name) {
    size_t col_idx = find_column_index(col_name);

    data_features.erase(data_features.begin() + col_idx);
    column_types.erase(column_types.begin() + col_idx);
    non_null_counts.erase(non_null_counts.begin() + col_idx);
    null_positions.erase(null_positions.begin() + col_idx);

    for (auto& row : data_values) {
        if (col_idx < row.size()) {
            row.erase(row.begin() + col_idx);
        }
    }
}

void vegaDataframe::drop_columns(const std::vector<std::string>& col_names) {
    for (const auto& col_name : col_names) {
        drop_column(col_name);
    }
}

void vegaDataframe::rename_column(const std::string& old_name, const std::string& new_name) {
    size_t col_idx = find_column_index(old_name);
    data_features[col_idx] = new_name;
}

void vegaDataframe::rename_columns(const std::map<std::string, std::string>& rename_map) {
    for (const auto& pair : rename_map) {
        rename_column(pair.first, pair.second);
    }
}

std::vector<std::string> vegaDataframe::columns() const {
    return data_features;
}

// ============= ROW OPERATIONS =============

vegaDataframe vegaDataframe::filter_rows(const std::string& col_name, const std::string& value) const {
    size_t col_idx = find_column_index(col_name);

    return filter_rows([col_idx, value](const std::vector<std::string>& row) {
        return col_idx < row.size() && row[col_idx] == value;
    });
}

vegaDataframe vegaDataframe::filter_rows(const std::function<bool(const std::vector<std::string>&)>& condition) const {
    vegaDataframe result;
    result.data_features = data_features;
    result.column_types = column_types;

    for (const auto& row : data_values) {
        if (condition(row)) {
            result.data_values.push_back(row);
        }
    }

    result.update_stats_after_modification();
    return result;
}

vegaDataframe vegaDataframe::query(const std::string& expression) const {
    // Simple query implementation - can be extended for complex expressions
    auto tokens = split_string(expression, ' ');
    if (tokens.size() >= 3) {
        std::string col_name = tokens[0];
        std::string op = tokens[1];
        std::string value = tokens[2];

        if (op == "==") {
            return filter_rows(col_name, value);
        }
        // Add more operators as needed
    }

    return *this; // Return copy if query cannot be parsed
}

void vegaDataframe::drop_row(size_t row_index) {
    if (row_index >= data_values.size())
        throw std::runtime_error("Row index out of range");

    data_values.erase(data_values.begin() + row_index);
    update_stats_after_modification();
}

void vegaDataframe::drop_rows(const std::vector<size_t>& row_indices) {
    std::vector<size_t> sorted_indices = row_indices;
    std::sort(sorted_indices.rbegin(), sorted_indices.rend()); // Sort in descending order

    for (size_t idx : sorted_indices) {
        drop_row(idx);
    }
}

vegaDataframe vegaDataframe::sample(size_t n, bool replace) const {
    vegaDataframe result;
    result.data_features = data_features;
    result.column_types = column_types;

    if (n >= data_values.size() && !replace) {
        result = *this;
        return result;
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    if (replace) {
        std::uniform_int_distribution<> dis(0, data_values.size() - 1);
        for (size_t i = 0; i < n; ++i) {
            size_t idx = dis(gen);
            result.data_values.push_back(data_values[idx]);
        }
    } else {
        std::vector<size_t> indices(data_values.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::ranges::shuffle(indices, gen);

        for (size_t i = 0; i < n; ++i) {
            result.data_values.push_back(data_values[indices[i]]);
        }
    }

    result.update_stats_after_modification();
    return result;
}

vegaDataframe vegaDataframe::nlargest(size_t n, const std::string& col_name) const {
    size_t col_idx = find_column_index(col_name);

    // Create vector of (value, row_index) pairs
    std::vector<std::pair<double, size_t>> values_with_indices;
    for (size_t i = 0; i < data_values.size(); ++i) {
        if (col_idx < data_values[i].size() && !data_values[i][col_idx].empty()) {
            try {
                double val = std::stod(data_values[i][col_idx]);
                values_with_indices.push_back({val, i});
            } catch (...) {}
        }
    }

    // Sort by value in descending order
    std::ranges::sort(values_with_indices,
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    // Create result dataframe
    vegaDataframe result;
    result.data_features = data_features;
    result.column_types = column_types;

    size_t count = std::min(n, values_with_indices.size());
    for (size_t i = 0; i < count; ++i) {
        size_t row_idx = values_with_indices[i].second;
        result.data_values.push_back(data_values[row_idx]);
    }

    result.update_stats_after_modification();
    return result;
}

vegaDataframe vegaDataframe::nsmallest(size_t n, const std::string& col_name) const {
    size_t col_idx = find_column_index(col_name);

    std::vector<std::pair<double, size_t>> values_with_indices;
    for (size_t i = 0; i < data_values.size(); ++i) {
        if (col_idx < data_values[i].size() && !data_values[i][col_idx].empty()) {
            try {
                double val = std::stod(data_values[i][col_idx]);
                values_with_indices.push_back({val, i});
            } catch (...) {}
        }
    }

    // Sort by value in ascending order
    std::ranges::sort(values_with_indices,
                      [](const auto& a, const auto& b) { return a.first < b.first; });

    vegaDataframe result;
    result.data_features = data_features;
    result.column_types = column_types;

    size_t count = std::min(n, values_with_indices.size());
    for (size_t i = 0; i < count; ++i) {
        size_t row_idx = values_with_indices[i].second;
        result.data_values.push_back(data_values[row_idx]);
    }

    result.update_stats_after_modification();
    return result;
}

// ============= INDEXING AND SELECTION =============

vegaDataframe vegaDataframe::loc(const std::vector<size_t>& rows, const std::vector<std::string>& cols) const {
    vegaDataframe result;

    // Set up columns
    for (const auto& col_name : cols) {
        result.data_features.push_back(col_name);
    }

    // Get column indices
    std::vector<size_t> col_indices;
    for (const auto& col_name : cols) {
        col_indices.push_back(find_column_index(col_name));
    }

    // Copy data types for selected columns
    for (size_t col_idx : col_indices) {
        result.column_types.push_back(column_types[col_idx]);
    }

    // Copy selected rows and columns
    for (size_t row_idx : rows) {
        if (row_idx < data_values.size()) {
            std::vector<std::string> new_row;
            for (size_t col_idx : col_indices) {
                if (col_idx < data_values[row_idx].size()) {
                    new_row.push_back(data_values[row_idx][col_idx]);
                } else {
                    new_row.push_back("");
                }
            }
            result.data_values.push_back(new_row);
        }
    }

    result.update_stats_after_modification();
    return result;
}

vegaDataframe vegaDataframe::iloc(const std::vector<size_t>& rows, const std::vector<size_t>& cols) const {
    vegaDataframe result;

    // Set up columns
    for (size_t col_idx : cols) {
        if (col_idx < data_features.size()) {
            result.data_features.push_back(data_features[col_idx]);
            result.column_types.push_back(column_types[col_idx]);
        }
    }

    // Copy selected rows and columns
    for (size_t row_idx : rows) {
        if (row_idx < data_values.size()) {
            std::vector<std::string> new_row;
            for (size_t col_idx : cols) {
                if (col_idx < data_values[row_idx].size()) {
                    new_row.push_back(data_values[row_idx][col_idx]);
                } else {
                    new_row.push_back("");
                }
            }
            result.data_values.push_back(new_row);
        }
    }

    result.update_stats_after_modification();
    return result;
}

std::string vegaDataframe::at(size_t row, const std::string& col) const {
    size_t col_idx = find_column_index(col);
    return iat(row, col_idx);
}

std::string vegaDataframe::iat(size_t row, size_t col) const {
    if (row >= data_values.size()) {
        throw std::runtime_error("Row index out of range");
    }
    if (col >= data_features.size()) {
        throw std::runtime_error("Column index out of range");
    }

    if (col < data_values[row].size()) {
        return data_values[row][col];
    } else {
        return "";
    }
}

// ============= STATISTICAL OPERATIONS =============

double vegaDataframe::mean(const std::string& col_name) const {
    size_t col_idx = find_column_index(col_name);

    if (column_types[col_idx] == DataType::STRING)
        throw std::runtime_error("Cannot compute mean for string column");

    double sum = 0;
    size_t count = 0;

    for (const auto& row : data_values) {
        if (col_idx < row.size() && !row[col_idx].empty()) {
            try {
                sum += std::stod(row[col_idx]);
                count++;
            } catch (...) {}
        }
    }

    if (count == 0) throw std::runtime_error("No valid values to compute mean");
    return sum / count;
}

double vegaDataframe::median(const std::string& col_name) const {
    size_t col_idx = find_column_index(col_name);

    if (column_types[col_idx] == DataType::STRING)
        throw std::runtime_error("Cannot compute median for string column");

    std::vector<double> values;
    for (const auto& row : data_values) {
        if (col_idx < row.size() && !row[col_idx].empty()) {
            try {
                values.push_back(std::stod(row[col_idx]));
            } catch (...) {}
        }
    }

    if (values.empty()) throw std::runtime_error("No valid values to compute median");

    std::ranges::sort(values);
    size_t n = values.size();

    if (n % 2 == 0) {
        return (values[n/2 - 1] + values[n/2]) / 2.0;
    } else {
        return values[n/2];
    }
}

std::string vegaDataframe::mode(const std::string& col_name) const {
    size_t col_idx = find_column_index(col_name);

    std::map<std::string, size_t> counts;
    for (const auto& row : data_values) {
        if (col_idx < row.size() && !row[col_idx].empty()) {
            counts[row[col_idx]]++;
        }
    }

    if (counts.empty()) throw std::runtime_error("No valid values to compute mode");

    auto max_elem = std::ranges::max_element(counts,
                                             [](const auto& a, const auto& b) { return a.second < b.second; });

    return max_elem->first;
}

double vegaDataframe::std_dev(const std::string& col_name) const {
    size_t col_idx = find_column_index(col_name);

    if (column_types[col_idx] == DataType::STRING)
        throw std::runtime_error("Cannot compute standard deviation for string column");

    double mean_val = mean(col_name);
    double sum_squared_diff = 0;
    size_t count = 0;

    for (const auto& row : data_values) {
        if (col_idx < row.size() && !row[col_idx].empty()) {
            try {
                double val = std::stod(row[col_idx]);
                sum_squared_diff += (val - mean_val) * (val - mean_val);
                count++;
            } catch (...) {}
        }
    }

    if (count <= 1) throw std::runtime_error("Need at least 2 values to compute standard deviation");
    return std::sqrt(sum_squared_diff / (count - 1));
}

double vegaDataframe::variance(const std::string& col_name) const {
    double std_val = std_dev(col_name);
    return std_val * std_val;
}

double vegaDataframe::min(const std::string& col_name) const {
    size_t col_idx = find_column_index(col_name);

    if (column_types[col_idx] == DataType::STRING)
        throw std::runtime_error("Cannot compute min for string column");

    double min_val = std::numeric_limits<double>::max();
    bool found = false;

    for (const auto& row : data_values) {
        if (col_idx < row.size() && !row[col_idx].empty()) {
            try {
                double val = std::stod(row[col_idx]);
                min_val = std::min(min_val, val);
                found = true;
            } catch (...) {}
        }
    }

    if (!found) throw std::runtime_error("No valid values to compute min");
    return min_val;
}

double vegaDataframe::max(const std::string& col_name) const {
    size_t col_idx = find_column_index(col_name);

    if (column_types[col_idx] == DataType::STRING)
        throw std::runtime_error("Cannot compute max for string column");

    double max_val = std::numeric_limits<double>::lowest();
    bool found = false;

    for (const auto& row : data_values) {
        if (col_idx < row.size() && !row[col_idx].empty()) {
            try {
                double val = std::stod(row[col_idx]);
                max_val = std::max(max_val, val);
                found = true;
            } catch (...) {}
        }
    }

    if (!found) throw std::runtime_error("No valid values to compute max");
    return max_val;
}

double vegaDataframe::sum(const std::string& col_name) const {
    size_t col_idx = find_column_index(col_name);

    if (column_types[col_idx] == DataType::STRING)
        throw std::runtime_error("Cannot compute sum for string column");

    double total = 0;
    for (const auto& row : data_values) {
        if (col_idx < row.size() && !row[col_idx].empty()) {
            try {
                total += std::stod(row[col_idx]);
            } catch (...) {}
        }
    }
    return total;
}

double vegaDataframe::prod(const std::string& col_name) const {
    size_t col_idx = find_column_index(col_name);

    if (column_types[col_idx] == DataType::STRING)
        throw std::runtime_error("Cannot compute product for string column");

    double product = 1.0;
    for (const auto& row : data_values) {
        if (col_idx < row.size() && !row[col_idx].empty()) {
            try {
                product *= std::stod(row[col_idx]);
            } catch (...) {}
        }
    }
    return product;
}

size_t vegaDataframe::count(const std::string& col_name) const {
    size_t col_idx = find_column_index(col_name);
    return non_null_counts[col_idx];
}

size_t vegaDataframe::nunique(const std::string& col_name) const {
    auto unique_vals = unique(col_name);
    return unique_vals.size();
}

std::map<std::string, size_t> vegaDataframe::value_counts(const std::string& col_name) const {
    size_t col_idx = find_column_index(col_name);

    std::map<std::string, size_t> counts;
    for (const auto& row : data_values) {
        if (col_idx < row.size() && !row[col_idx].empty()) {
            counts[row[col_idx]]++;
        }
    }
    return counts;
}

std::vector<double> vegaDataframe::quantile(const std::string& col_name, const std::vector<double>& q) const {
    size_t col_idx = find_column_index(col_name);

    if (column_types[col_idx] == DataType::STRING)
        throw std::runtime_error("Cannot compute quantiles for string column");

    std::vector<double> values;
    for (const auto& row : data_values) {
        if (col_idx < row.size() && !row[col_idx].empty()) {
            try {
                values.push_back(std::stod(row[col_idx]));
            } catch (...) {}
        }
    }

    if (values.empty()) throw std::runtime_error("No valid values to compute quantiles");

    std::ranges::sort(values);
    std::vector<double> result;

    for (double quantile : q) {
        if (quantile < 0.0 || quantile > 1.0) {
            throw std::runtime_error("Quantile must be between 0 and 1");
        }

        double pos = quantile * (values.size() - 1);
        size_t lower = static_cast<size_t>(std::floor(pos));
        size_t upper = static_cast<size_t>(std::ceil(pos));

        if (lower == upper) {
            result.push_back(values[lower]);
        } else {
            double weight = pos - lower;
            result.push_back(values[lower] * (1 - weight) + values[upper] * weight);
        }
    }

    return result;
}

std::map<std::string, double> vegaDataframe::corr() const {
    std::map<std::string, double> correlations;

    // Get all numeric columns
    std::vector<std::string> numeric_cols;
    for (size_t i = 0; i < data_features.size(); ++i) {
        if (column_types[i] != DataType::STRING) {
            numeric_cols.push_back(data_features[i]);
        }
    }

    // Calculate correlation between each pair of numeric columns
    for (size_t i = 0; i < numeric_cols.size(); ++i) {
        for (size_t j = i; j < numeric_cols.size(); ++j) {
            std::string key = numeric_cols[i] + "_" + numeric_cols[j];

            if (i == j) {
                correlations[key] = 1.0;
            } else {
                // Calculate Pearson correlation coefficient
                try {
                    std::vector<double> x_vals, y_vals;
                    size_t x_idx = find_column_index(numeric_cols[i]);
                    size_t y_idx = find_column_index(numeric_cols[j]);

                    for (const auto& row : data_values) {
                        if (x_idx < row.size() && y_idx < row.size() &&
                            !row[x_idx].empty() && !row[y_idx].empty()) {
                            try {
                                x_vals.push_back(std::stod(row[x_idx]));
                                y_vals.push_back(std::stod(row[y_idx]));
                            } catch (...) {}
                        }
                    }

                    if (x_vals.size() > 1) {
                        double x_mean = std::accumulate(x_vals.begin(), x_vals.end(), 0.0) / x_vals.size();
                        double y_mean = std::accumulate(y_vals.begin(), y_vals.end(), 0.0) / y_vals.size();

                        double numerator = 0.0, x_sq_sum = 0.0, y_sq_sum = 0.0;
                        for (size_t k = 0; k < x_vals.size(); ++k) {
                            double x_diff = x_vals[k] - x_mean;
                            double y_diff = y_vals[k] - y_mean;
                            numerator += x_diff * y_diff;
                            x_sq_sum += x_diff * x_diff;
                            y_sq_sum += y_diff * y_diff;
                        }

                        double denominator = std::sqrt(x_sq_sum * y_sq_sum);
                        correlations[key] = (denominator > 0) ? numerator / denominator : 0.0;
                    } else {
                        correlations[key] = 0.0;
                    }
                } catch (...) {
                    correlations[key] = 0.0;
                }
            }
        }
    }

    return correlations;
}

std::map<std::string, double> vegaDataframe::cov() const {
    std::map<std::string, double> covariances;

    std::vector<std::string> numeric_cols;
    for (size_t i = 0; i < data_features.size(); ++i) {
        if (column_types[i] != DataType::STRING) {
            numeric_cols.push_back(data_features[i]);
        }
    }

    for (size_t i = 0; i < numeric_cols.size(); ++i) {
        for (size_t j = i; j < numeric_cols.size(); ++j) {
            std::string key = numeric_cols[i] + "_" + numeric_cols[j];

            try {
                std::vector<double> x_vals, y_vals;
                size_t x_idx = find_column_index(numeric_cols[i]);
                size_t y_idx = find_column_index(numeric_cols[j]);

                for (const auto& row : data_values) {
                    if (x_idx < row.size() && y_idx < row.size() &&
                        !row[x_idx].empty() && !row[y_idx].empty()) {
                        try {
                            x_vals.push_back(std::stod(row[x_idx]));
                            y_vals.push_back(std::stod(row[y_idx]));
                        } catch (...) {}
                    }
                }

                if (x_vals.size() > 1) {
                    double x_mean = std::accumulate(x_vals.begin(), x_vals.end(), 0.0) / x_vals.size();
                    double y_mean = std::accumulate(y_vals.begin(), y_vals.end(), 0.0) / y_vals.size();

                    double covariance = 0.0;
                    for (size_t k = 0; k < x_vals.size(); ++k) {
                        covariance += (x_vals[k] - x_mean) * (y_vals[k] - y_mean);
                    }
                    covariances[key] = covariance / (x_vals.size() - 1);
                } else {
                    covariances[key] = 0.0;
                }
            } catch (...) {
                covariances[key] = 0.0;
            }
        }
    }

    return covariances;
}

// ============= MISSING DATA HANDLING =============

vegaDataframe vegaDataframe::dropna(const std::string& how) const {
    vegaDataframe result;
    result.data_features = data_features;
    result.column_types = column_types;

    for (const auto& row : data_values) {
        bool has_null = false;
        bool all_null = true;

        for (const auto& cell : row) {
            if (cell.empty()) {
                has_null = true;
            } else {
                all_null = false;
            }
        }

        if (how == "any" && !has_null) {
            result.data_values.push_back(row);
        } else if (how == "all" && !all_null) {
            result.data_values.push_back(row);
        }
    }

    result.update_stats_after_modification();
    return result;
}

void vegaDataframe::fillna_with_imputer(const std::string& col_name, Imputer& imputer) {
    imputer.impute(*this, col_name);
}

void vegaDataframe::fillna_value(const std::string& col_name, const std::string& value) {
    size_t col_idx = find_column_index(col_name);

    for (auto& row : data_values) {
        if (col_idx < row.size() && row[col_idx].empty()) {
            row[col_idx] = value;
        }
    }

    update_stats_after_modification();
}

void vegaDataframe::fillna_method(const std::string& col_name, const std::string& method) {
    size_t col_idx = find_column_index(col_name);

    if (method == "ffill" || method == "pad") {
        std::string last_valid = "";
        for (auto& row : data_values) {
            if (col_idx < row.size()) {
                if (row[col_idx].empty()) {
                    if (!last_valid.empty()) {
                        row[col_idx] = last_valid;
                    }
                } else {
                    last_valid = row[col_idx];
                }
            }
        }
    } else if (method == "bfill" || method == "backfill") {
        std::string next_valid = "";
        for (int i = static_cast<int>(data_values.size()) - 1; i >= 0; --i) {
            if (col_idx < data_values[i].size()) {
                if (data_values[i][col_idx].empty()) {
                    if (!next_valid.empty()) {
                        data_values[i][col_idx] = next_valid;
                    }
                } else {
                    next_valid = data_values[i][col_idx];
                }
            }
        }
    }

    update_stats_after_modification();
}

vegaDataframe vegaDataframe::interpolate(const std::string& col_name, const std::string& method) const {
    vegaDataframe result = *this;
    size_t col_idx = find_column_index(col_name);

    if (column_types[col_idx] == DataType::STRING) {
        throw std::runtime_error("Cannot interpolate string column");
    }

    if (method == "linear") {
        // Simple linear interpolation
        for (size_t i = 1; i < result.data_values.size() - 1; ++i) {
            if (col_idx < result.data_values[i].size() && result.data_values[i][col_idx].empty()) {
                // Find previous and next non-empty values
                double prev_val = 0, next_val = 0;
                size_t prev_idx = 0, next_idx = 0;
                bool found_prev = false, found_next = false;

                // Find previous value
                for (int j = static_cast<int>(i) - 1; j >= 0; --j) {
                    if (col_idx < result.data_values[j].size() && !result.data_values[j][col_idx].empty()) {
                        try {
                            prev_val = std::stod(result.data_values[j][col_idx]);
                            prev_idx = j;
                            found_prev = true;
                            break;
                        } catch (...) {}
                    }
                }

                // Find next value
                for (size_t j = i + 1; j < result.data_values.size(); ++j) {
                    if (col_idx < result.data_values[j].size() && !result.data_values[j][col_idx].empty()) {
                        try {
                            next_val = std::stod(result.data_values[j][col_idx]);
                            next_idx = j;
                            found_next = true;
                            break;
                        } catch (...) {}
                    }
                }

                if (found_prev && found_next) {
                    double ratio = static_cast<double>(i - prev_idx) / (next_idx - prev_idx);
                    double interpolated = prev_val + ratio * (next_val - prev_val);
                    result.data_values[i][col_idx] = std::to_string(interpolated);
                }
            }
        }
    }

    result.update_stats_after_modification();
    return result;
}

// ============= SORTING OPERATIONS =============

void vegaDataframe::sort_values(const std::string& col_name, bool ascending) {
    size_t col_idx = find_column_index(col_name);

    std::ranges::sort(data_values,
                      [col_idx, ascending](const std::vector<std::string>& a, const std::vector<std::string>& b) {
                          if (col_idx >= a.size() || col_idx >= b.size()) return false;

                          if (ascending) {
                              return a[col_idx] < b[col_idx];
                          } else {
                              return a[col_idx] > b[col_idx];
                          }
                      });

    update_stats_after_modification();
}

void vegaDataframe::sort_values(const std::vector<std::string>& col_names, const std::vector<bool>& ascending) {
    if (col_names.size() != ascending.size()) {
        throw std::runtime_error("Column names and ascending vectors must have same size");
    }

    std::vector<size_t> col_indices;
    for (const auto& col_name : col_names) {
        col_indices.push_back(find_column_index(col_name));
    }

    std::ranges::sort(data_values,
                      [&col_indices, &ascending](const std::vector<std::string>& a, const std::vector<std::string>& b) {
                          for (size_t i = 0; i < col_indices.size(); ++i) {
                              size_t col_idx = col_indices[i];
                              if (col_idx >= a.size() || col_idx >= b.size()) continue;

                              if (a[col_idx] != b[col_idx]) {
                                  if (ascending[i]) {
                                      return a[col_idx] < b[col_idx];
                                  } else {
                                      return a[col_idx] > b[col_idx];
                                  }
                              }
                          }
                          return false;
                      });

    update_stats_after_modification();
}

void vegaDataframe::sort_index(bool ascending) {
    // Since we don't have explicit index, sort by row order
    if (!ascending) {
        std::ranges::reverse(data_values);
    }
    update_stats_after_modification();
}

vegaDataframe vegaDataframe::rank(const std::string& col_name, const std::string& method) const {
    size_t col_idx = find_column_index(col_name);

    if (column_types[col_idx] == DataType::STRING) {
        throw std::runtime_error("Cannot rank string column");
    }

    // Create vector of (value, original_index) pairs
    std::vector<std::pair<double, size_t>> values_with_indices;
    for (size_t i = 0; i < data_values.size(); ++i) {
        if (col_idx < data_values[i].size() && !data_values[i][col_idx].empty()) {
            try {
                double val = std::stod(data_values[i][col_idx]);
                values_with_indices.push_back({val, i});
            } catch (...) {}
        }
    }

    // Sort by value
    std::ranges::sort(values_with_indices);

    // Create result dataframe
    vegaDataframe result = *this;
    std::vector<std::string> rank_column(data_values.size(), "");

    // Assign ranks
    for (size_t i = 0; i < values_with_indices.size(); ++i) {
        size_t original_idx = values_with_indices[i].second;
        rank_column[original_idx] = std::to_string(i + 1);
    }

    result.add_column(col_name + "_rank", rank_column);
    return result;
}

// Continue with the remaining implementations...
// This is Part 1 of the implementation file. Due to length constraints,
// I'll need to create additional parts for the remaining functions.

// ============= IMPUTER IMPLEMENTATIONS =============

void MeanImputer::impute(vegaDataframe& df, const std::string& column) {
    size_t col_idx = df.find_column_index(column);

    if (df.column_types[col_idx] == DataType::STRING)
        throw std::runtime_error("Mean imputation only applicable to numeric columns");

    double sum = 0.0;
    size_t count = 0;
    for (const auto& row : df.data_values) {
        if (col_idx < row.size() && !row[col_idx].empty()) {
            try {
                sum += std::stod(row[col_idx]);
                count++;
            } catch (...) {}
        }
    }

    if (count == 0) return;

    double mean = sum / count;
    std::string mean_str = std::to_string(mean);

    for (auto& row : df.data_values) {
        if (col_idx < row.size() && row[col_idx].empty()) {
            row[col_idx] = mean_str;
        }
    }

    df.null_positions[col_idx].clear();
    df.non_null_counts[col_idx] = df.data_values.size();

    std::cout << "Mean imputation performed on column '" << column << "' with value: " << mean_str << "\n";
}

void MedianImputer::impute(vegaDataframe& df, const std::string& column) {
    size_t col_idx = df.find_column_index(column);

    if (df.column_types[col_idx] == DataType::STRING)
        throw std::runtime_error("Median imputation only applicable to numeric columns");

    std::vector<double> values;
    for (const auto& row : df.data_values) {
        if (col_idx < row.size() && !row[col_idx].empty()) {
            try {
                values.push_back(std::stod(row[col_idx]));
            } catch (...) {}
        }
    }

    if (values.empty()) return;

    std::ranges::sort(values);
    double median;
    size_t n = values.size();

    if (n % 2 == 0) {
        median = (values[n/2 - 1] + values[n/2]) / 2.0;
    } else {
        median = values[n/2];
    }

    std::string median_str = std::to_string(median);

    for (auto& row : df.data_values) {
        if (col_idx < row.size() && row[col_idx].empty()) {
            row[col_idx] = median_str;
        }
    }

    df.null_positions[col_idx].clear();
    df.non_null_counts[col_idx] = df.data_values.size();

    std::cout << "Median imputation performed on column '" << column << "' with value: " << median_str << "\n";
}

void ModeImputer::impute(vegaDataframe& df, const std::string& column) {
    size_t col_idx = df.find_column_index(column);

    std::map<std::string, size_t> counts;
    for (const auto& row : df.data_values) {
        if (col_idx < row.size() && !row[col_idx].empty()) {
            counts[row[col_idx]]++;
        }
    }

    if (counts.empty()) return;

    auto max_elem = std::ranges::max_element(counts,
                                             [](const auto& a, const auto& b) { return a.second < b.second; });

    std::string mode_value = max_elem->first;

    for (auto& row : df.data_values) {
        if (col_idx < row.size() && row[col_idx].empty()) {
            row[col_idx] = mode_value;
        }
    }

    df.null_positions[col_idx].clear();
    df.non_null_counts[col_idx] = df.data_values.size();

    std::cout << "Mode imputation performed on column '" << column << "' with value: '" << mode_value << "'\n";
}

void ConstantImputer::impute(vegaDataframe& df, const std::string& column) {
    size_t col_idx = df.find_column_index(column);

    for (auto& row : df.data_values) {
        if (col_idx < row.size() && row[col_idx].empty()) {
            row[col_idx] = fill_value;
        }
    }

    df.null_positions[col_idx].clear();
    df.non_null_counts[col_idx] = df.data_values.size();

    std::cout << "Constant imputation performed on column '" << column << "' with value: '" << fill_value << "'\n";
}

void ForwardFillImputer::impute(vegaDataframe& df, const std::string& column) {
    size_t col_idx = df.find_column_index(column);

    std::string last_valid_value = "";

    for (auto& row : df.data_values) {
        if (col_idx < row.size()) {
            if (row[col_idx].empty()) {
                if (!last_valid_value.empty()) {
                    row[col_idx] = last_valid_value;
                }
            } else {
                last_valid_value = row[col_idx];
            }
        }
    }

    df.update_stats_after_modification();
    std::cout << "Forward fill imputation performed on column '" << column << "'\n";
}

void BackwardFillImputer::impute(vegaDataframe& df, const std::string& column) {
    size_t col_idx = df.find_column_index(column);

    std::vector<std::string> next_valid(df.data_values.size(), "");
    std::string next_valid_value = "";

    for (int i = static_cast<int>(df.data_values.size()) - 1; i >= 0; --i) {
        if (col_idx < df.data_values[i].size()) {
            if (!df.data_values[i][col_idx].empty()) {
                next_valid_value = df.data_values[i][col_idx];
            }
            next_valid[i] = next_valid_value;
        }
    }

    for (size_t i = 0; i < df.data_values.size(); ++i) {
        if (col_idx < df.data_values[i].size() && df.data_values[i][col_idx].empty()) {
            if (!next_valid[i].empty()) {
                df.data_values[i][col_idx] = next_valid[i];
            }
        }
    }

    df.update_stats_after_modification();
    std::cout << "Backward fill imputation performed on column '" << column << "'\n";
}

void LinearInterpolationImputer::impute(vegaDataframe& df, const std::string& column) {
    size_t col_idx = df.find_column_index(column);

    if (df.column_types[col_idx] == DataType::STRING) {
        throw std::runtime_error("Cannot interpolate string column");
    }

    for (size_t i = 1; i < df.data_values.size() - 1; ++i) {
        if (col_idx < df.data_values[i].size() && df.data_values[i][col_idx].empty()) {
            double prev_val = 0, next_val = 0;
            size_t prev_idx = 0, next_idx = 0;
            bool found_prev = false, found_next = false;

            // Find previous value
            for (int j = static_cast<int>(i) - 1; j >= 0; --j) {
                if (col_idx < df.data_values[j].size() && !df.data_values[j][col_idx].empty()) {
                    try {
                        prev_val = std::stod(df.data_values[j][col_idx]);
                        prev_idx = j;
                        found_prev = true;
                        break;
                    } catch (...) {}
                }
            }

            // Find next value
            for (size_t j = i + 1; j < df.data_values.size(); ++j) {
                if (col_idx < df.data_values[j].size() && !df.data_values[j][col_idx].empty()) {
                    try {
                        next_val = std::stod(df.data_values[j][col_idx]);
                        next_idx = j;
                        found_next = true;
                        break;
                    } catch (...) {}
                }
            }

            if (found_prev && found_next) {
                double ratio = static_cast<double>(i - prev_idx) / (next_idx - prev_idx);
                double interpolated = prev_val + ratio * (next_val - prev_val);
                df.data_values[i][col_idx] = std::to_string(interpolated);
            }
        }
    }

    df.update_stats_after_modification();
    std::cout << "Linear interpolation performed on column '" << column << "'\n";
}

// ============= GROUPING AND AGGREGATION =============

std::map<std::string, vegaDataframe> vegaDataframe::groupby(const std::string& col_name) const {
    size_t col_idx = find_column_index(col_name);
    std::map<std::string, vegaDataframe> groups;

    for (const auto& row : data_values) {
        if (col_idx < row.size()) {
            std::string key = row[col_idx];
            if (!groups.contains(key)) {
                groups[key].data_features = data_features;
                groups[key].column_types = column_types;
            }
            groups[key].data_values.push_back(row);
        }
    }

    for (auto &val: groups | std::views::values) {
        val.update_stats_after_modification();
    }

    return groups;
}

std::map<std::vector<std::string>, vegaDataframe> vegaDataframe::groupby(const std::vector<std::string>& col_names) const {
    std::vector<size_t> col_indices;
    for (const auto& col_name : col_names) {
        col_indices.push_back(find_column_index(col_name));
    }

    std::map<std::vector<std::string>, vegaDataframe> groups;

    for (const auto& row : data_values) {
        std::vector<std::string> key;
        for (size_t col_idx : col_indices) {
            if (col_idx < row.size()) {
                key.push_back(row[col_idx]);
            } else {
                key.push_back("");
            }
        }

        if (!groups.contains(key)) {
            groups[key].data_features = data_features;
            groups[key].column_types = column_types;
        }
        groups[key].data_values.push_back(row);
    }

    for (auto &val: groups | std::views::values) {
        val.update_stats_after_modification();
    }

    return groups;
}

vegaDataframe vegaDataframe::aggregate(const std::map<std::string, std::string>& agg_funcs) const {
    vegaDataframe result;

    // Create result columns
    for (const auto& pair : agg_funcs) {
        result.data_features.push_back(pair.first + "_" + pair.second);
        result.column_types.push_back(DataType::FLOAT);
    }

    // Calculate aggregations
    std::vector<std::string> agg_row;
    for (const auto& pair : agg_funcs) {
        const std::string& col_name = pair.first;
        const std::string& func_name = pair.second;

        try {
            double result_val = 0.0;
            if (func_name == "mean") {
                result_val = mean(col_name);
            } else if (func_name == "sum") {
                result_val = sum(col_name);
            } else if (func_name == "min") {
                result_val = min(col_name);
            } else if (func_name == "max") {
                result_val = max(col_name);
            } else if (func_name == "count") {
                result_val = static_cast<double>(count(col_name));
            } else if (func_name == "std") {
                result_val = std_dev(col_name);
            }
            agg_row.push_back(std::to_string(result_val));
        } catch (...) {
            agg_row.push_back("NaN");
        }
    }

    result.data_values.push_back(agg_row);
    result.update_stats_after_modification();
    return result;
}

// ============= DATA TRANSFORMATION =============

void vegaDataframe::label_encode(const std::string& col_name) {
    size_t col_idx = find_column_index(col_name);

    if (column_types[col_idx] != DataType::STRING)
        throw std::runtime_error("Label encoding applies only to string columns");

    std::unordered_map<std::string, int> label_map;
    int next_label = 0;

    for (auto& row : data_values) {
        if (col_idx < row.size() && !row[col_idx].empty()) {
            if (!label_map.contains(row[col_idx])) {
                label_map[row[col_idx]] = next_label++;
            }
            row[col_idx] = std::to_string(label_map[row[col_idx]]);
        }
    }

    column_types[col_idx] = DataType::INT;
    std::cout << "Label encoding applied on column '" << col_name << "' with " << next_label << " categories.\n";
}

vegaDataframe vegaDataframe::one_hot_encode(const std::string& col_name) const {
    size_t col_idx = find_column_index(col_name);

    if (column_types[col_idx] != DataType::STRING)
        throw std::runtime_error("One-hot encoding applies only to string columns");

    std::set<std::string> unique_values;
    for (const auto& row : data_values) {
        if (col_idx < row.size() && !row[col_idx].empty()) {
            unique_values.insert(row[col_idx]);
        }
    }

    vegaDataframe result = *this;

    for (const std::string& value : unique_values) {
        std::string new_col_name = col_name + "_" + value;
        std::vector<std::string> new_col_values;

        for (const auto& row : data_values) {
            if (col_idx < row.size() && row[col_idx] == value) {
                new_col_values.push_back("1");
            } else {
                new_col_values.push_back("0");
            }
        }

        result.add_column(new_col_name, new_col_values);
    }

    result.drop_column(col_name);
    return result;
}

vegaDataframe vegaDataframe::get_dummies(const std::vector<std::string>& col_names) const {
    vegaDataframe result = *this;

    for (const auto& col_name : col_names) {
        result = result.one_hot_encode(col_name);
    }

    return result;
}

void vegaDataframe::apply_function(const std::string& col_name, const std::function<std::string(const std::string&)>& func) {
    size_t col_idx = find_column_index(col_name);

    for (auto& row : data_values) {
        if (col_idx < row.size()) {
            row[col_idx] = func(row[col_idx]);
        }
    }

    update_stats_after_modification();
}

vegaDataframe vegaDataframe::map_values(const std::string& col_name, const std::map<std::string, std::string>& mapping) const {
    vegaDataframe result = *this;
    size_t col_idx = find_column_index(col_name);

    for (auto& row : result.data_values) {
        if (col_idx < row.size()) {
            auto it = mapping.find(row[col_idx]);
            if (it != mapping.end()) {
                row[col_idx] = it->second;
            }
        }
    }

    result.update_stats_after_modification();
    return result;
}

// ============= STRING OPERATIONS =============

vegaDataframe vegaDataframe::str_contains(const std::string& col_name, const std::string& pattern) const {
    vegaDataframe result = *this;
    size_t col_idx = find_column_index(col_name);

    std::vector<std::string> contains_col;
    for (const auto& row : data_values) {
        if (col_idx < row.size()) {
            bool contains = row[col_idx].find(pattern) != std::string::npos;
            contains_col.push_back(contains ? "True" : "False");
        } else {
            contains_col.push_back("False");
        }
    }

    result.add_column(col_name + "_contains", contains_col);
    return result;
}

vegaDataframe vegaDataframe::str_startswith(const std::string& col_name, const std::string& prefix) const {
    vegaDataframe result = *this;
    size_t col_idx = find_column_index(col_name);

    std::vector<std::string> startswith_col;
    for (const auto& row : data_values) {
        if (col_idx < row.size()) {
            bool starts = row[col_idx].substr(0, prefix.length()) == prefix;
            startswith_col.push_back(starts ? "True" : "False");
        } else {
            startswith_col.push_back("False");
        }
    }

    result.add_column(col_name + "_startswith", startswith_col);
    return result;
}

vegaDataframe vegaDataframe::str_endswith(const std::string& col_name, const std::string& suffix) const {
    vegaDataframe result = *this;
    size_t col_idx = find_column_index(col_name);

    std::vector<std::string> endswith_col;
    for (const auto& row : data_values) {
        if (col_idx < row.size()) {
            bool ends = false;
            if (row[col_idx].length() >= suffix.length()) {
                ends = row[col_idx].substr(row[col_idx].length() - suffix.length()) == suffix;
            }
            endswith_col.push_back(ends ? "True" : "False");
        } else {
            endswith_col.push_back("False");
        }
    }

    result.add_column(col_name + "_endswith", endswith_col);
    return result;
}

vegaDataframe vegaDataframe::str_replace(const std::string& col_name, const std::string& pattern, const std::string& replacement) const {
    vegaDataframe result = *this;
    size_t col_idx = find_column_index(col_name);

    for (auto& row : result.data_values) {
        if (col_idx < row.size()) {
            std::string& cell = row[col_idx];
            size_t pos = 0;
            while ((pos = cell.find(pattern, pos)) != std::string::npos) {
                cell.replace(pos, pattern.length(), replacement);
                pos += replacement.length();
            }
        }
    }

    return result;
}

vegaDataframe vegaDataframe::str_upper(const std::string& col_name) const {
    vegaDataframe result = *this;
    size_t col_idx = find_column_index(col_name);

    for (auto& row : result.data_values) {
        if (col_idx < row.size()) {
            std::ranges::transform(row[col_idx], row[col_idx].begin(), ::toupper);
        }
    }

    return result;
}

vegaDataframe vegaDataframe::str_lower(const std::string& col_name) const {
    vegaDataframe result = *this;
    size_t col_idx = find_column_index(col_name);

    for (auto& row : result.data_values) {
        if (col_idx < row.size()) {
            std::ranges::transform(row[col_idx], row[col_idx].begin(), ::tolower);
        }
    }

    return result;
}

vegaDataframe vegaDataframe::str_strip(const std::string& col_name) const {
    vegaDataframe result = *this;
    size_t col_idx = find_column_index(col_name);

    for (auto& row : result.data_values) {
        if (col_idx < row.size()) {
            row[col_idx] = trim_whitespace(row[col_idx]);
        }
    }

    return result;
}

std::vector<size_t> vegaDataframe::str_len(const std::string& col_name) const {
    size_t col_idx = find_column_index(col_name);

    std::vector<size_t> lengths;
    for (const auto& row : data_values) {
        if (col_idx < row.size()) {
            lengths.push_back(row[col_idx].length());
        } else {
            lengths.push_back(0);
        }
    }

    return lengths;
}

// ============= MERGING AND JOINING =============

vegaDataframe vegaDataframe::merge(const vegaDataframe& other, const std::string& left_col, const std::string& right_col, const std::string& how) const {
    size_t left_col_idx = find_column_index(left_col);
    size_t right_col_idx = other.find_column_index(right_col);

    vegaDataframe result;

    // Combine feature names
    result.data_features = data_features;
    for (const auto& feature : other.data_features) {
        if (feature != right_col) {  // Avoid duplicate join column
            result.data_features.push_back(feature);
        }
    }

    // Combine column types
    result.column_types = column_types;
    for (size_t i = 0; i < other.column_types.size(); ++i) {
        if (i != right_col_idx) {
            result.column_types.push_back(other.column_types[i]);
        }
    }

    if (how == "inner") {
        // Inner join
        for (const auto& left_row : data_values) {
            if (left_col_idx < left_row.size()) {
                std::string join_key = left_row[left_col_idx];

                for (const auto& right_row : other.data_values) {
                    if (right_col_idx < right_row.size() && right_row[right_col_idx] == join_key) {
                        std::vector<std::string> merged_row = left_row;

                        for (size_t i = 0; i < right_row.size(); ++i) {
                            if (i != right_col_idx) {
                                merged_row.push_back(right_row[i]);
                            }
                        }

                        result.data_values.push_back(merged_row);
                    }
                }
            }
        }
    } else if (how == "left") {
        // Left join
        for (const auto& left_row : data_values) {
            std::string join_key = (left_col_idx < left_row.size()) ? left_row[left_col_idx] : "";
            bool found_match = false;

            for (const auto& right_row : other.data_values) {
                if (right_col_idx < right_row.size() && right_row[right_col_idx] == join_key) {
                    std::vector<std::string> merged_row = left_row;

                    for (size_t i = 0; i < right_row.size(); ++i) {
                        if (i != right_col_idx) {
                            merged_row.push_back(right_row[i]);
                        }
                    }

                    result.data_values.push_back(merged_row);
                    found_match = true;
                }
            }

            if (!found_match) {
                std::vector<std::string> merged_row = left_row;
                // Add empty values for right table columns
                for (size_t i = 0; i < other.data_features.size(); ++i) {
                    if (i != right_col_idx) {
                        merged_row.push_back("");
                    }
                }
                result.data_values.push_back(merged_row);
            }
        }
    }
    // Add more join types (right, outer) as needed

    result.update_stats_after_modification();
    return result;
}

vegaDataframe vegaDataframe::merge(const vegaDataframe& other, const std::vector<std::string>& on, const std::string& how) const {
    // Multi-column merge implementation
    std::vector<size_t> left_col_indices, right_col_indices;

    for (const auto& col_name : on) {
        left_col_indices.push_back(find_column_index(col_name));
        right_col_indices.push_back(other.find_column_index(col_name));
    }

    vegaDataframe result;
    result.data_features = data_features;

    // Add columns from right table (excluding join columns)
    for (size_t i = 0; i < other.data_features.size(); ++i) {
        if (std::ranges::find(right_col_indices, i) == right_col_indices.end()) {
            result.data_features.push_back(other.data_features[i]);
        }
    }

    result.column_types = column_types;
    for (size_t i = 0; i < other.column_types.size(); ++i) {
        if (std::ranges::find(right_col_indices, i) == right_col_indices.end()) {
            result.column_types.push_back(other.column_types[i]);
        }
    }

    if (how == "inner") {
        for (const auto& left_row : data_values) {
            std::vector<std::string> left_key;
            for (size_t col_idx : left_col_indices) {
                if (col_idx < left_row.size()) {
                    left_key.push_back(left_row[col_idx]);
                } else {
                    left_key.push_back("");
                }
            }

            for (const auto& right_row : other.data_values) {
                std::vector<std::string> right_key;
                for (size_t col_idx : right_col_indices) {
                    if (col_idx < right_row.size()) {
                        right_key.push_back(right_row[col_idx]);
                    } else {
                        right_key.push_back("");
                    }
                }

                if (left_key == right_key) {
                    std::vector<std::string> merged_row = left_row;

                    for (size_t i = 0; i < right_row.size(); ++i) {
                        if (std::ranges::find(right_col_indices, i) == right_col_indices.end()) {
                            merged_row.push_back(right_row[i]);
                        }
                    }

                    result.data_values.push_back(merged_row);
                }
            }
        }
    }

    result.update_stats_after_modification();
    return result;
}

vegaDataframe vegaDataframe::concat(const std::vector<vegaDataframe>& dataframes, int axis, bool ignore_index) {
    if (dataframes.empty()) {
        return vegaDataframe();
    }

    vegaDataframe result = dataframes[0];

    if (axis == 0) {
        // Concatenate rows (vertically)
        for (size_t i = 1; i < dataframes.size(); ++i) {
            const auto& df = dataframes[i];

            // Check if columns match
            if (df.data_features != result.data_features) {
                throw std::runtime_error("Column names don't match for vertical concatenation");
            }

            // Append rows
            for (const auto& row : df.data_values) {
                result.data_values.push_back(row);
            }
        }
    } else if (axis == 1) {
        // Concatenate columns (horizontally)
        for (size_t i = 1; i < dataframes.size(); ++i) {
            const auto& df = dataframes[i];

            // Check if row counts match
            if (df.data_values.size() != result.data_values.size()) {
                throw std::runtime_error("Row counts don't match for horizontal concatenation");
            }

            // Append column names
            for (const auto& feature : df.data_features) {
                result.data_features.push_back(feature);
            }

            // Append column types
            for (const auto& type : df.column_types) {
                result.column_types.push_back(type);
            }

            // Append column data
            for (size_t row_idx = 0; row_idx < result.data_values.size(); ++row_idx) {
                for (size_t col_idx = 0; col_idx < df.data_values[row_idx].size(); ++col_idx) {
                    result.data_values[row_idx].push_back(df.data_values[row_idx][col_idx]);
                }
            }
        }
    }

    result.update_stats_after_modification();
    return result;
}

vegaDataframe vegaDataframe::join(const vegaDataframe& other, const std::string& how) const {
    // Simple index-based join (assumes matching row indices)
    vegaDataframe result = *this;

    // Add columns from other dataframe
    for (const auto& feature : other.data_features) {
        result.data_features.push_back(feature);
    }

    for (const auto& type : other.column_types) {
        result.column_types.push_back(type);
    }

    // Join data row by row
    size_t min_rows = std::min(data_values.size(), other.data_values.size());

    for (size_t i = 0; i < min_rows; ++i) {
        for (size_t j = 0; j < other.data_values[i].size(); ++j) {
            result.data_values[i].push_back(other.data_values[i][j]);
        }
    }

    // Handle remaining rows based on join type
    if (how == "left" && data_values.size() > other.data_values.size()) {
        for (size_t i = min_rows; i < data_values.size(); ++i) {
            for (size_t j = 0; j < other.data_features.size(); ++j) {
                result.data_values[i].push_back("");
            }
        }
    }

    result.update_stats_after_modification();
    return result;
}

// ============= DUPLICATE HANDLING =============

std::vector<bool> vegaDataframe::duplicated(const std::vector<std::string>& subset, bool keep_first) const {
    std::vector<bool> is_duplicate(data_values.size(), false);
    std::set<std::vector<std::string>> seen_combinations;

    std::vector<size_t> check_columns;
    if (subset.empty()) {
        // Check all columns
        for (size_t i = 0; i < data_features.size(); ++i) {
            check_columns.push_back(i);
        }
    } else {
        // Check specified columns
        for (const auto& col_name : subset) {
            check_columns.push_back(find_column_index(col_name));
        }
    }

    for (size_t row_idx = 0; row_idx < data_values.size(); ++row_idx) {
        const auto& row = data_values[row_idx];
        std::vector<std::string> key;

        for (size_t col_idx : check_columns) {
            if (col_idx < row.size()) {
                key.push_back(row[col_idx]);
            } else {
                key.push_back("");
            }
        }

        if (seen_combinations.contains(key)) {
            is_duplicate[row_idx] = true;
        } else {
            seen_combinations.insert(key);
        }
    }

    if (!keep_first) {
        // Mark first occurrence as duplicate instead of subsequent ones
        seen_combinations.clear();
        std::fill(is_duplicate.begin(), is_duplicate.end(), false);

        for (int row_idx = static_cast<int>(data_values.size()) - 1; row_idx >= 0; --row_idx) {
            const auto& row = data_values[row_idx];
            std::vector<std::string> key;

            for (size_t col_idx : check_columns) {
                if (col_idx < row.size()) {
                    key.push_back(row[col_idx]);
                } else {
                    key.emplace_back("");
                }
            }

            if (seen_combinations.contains(key)) {
                is_duplicate[row_idx] = true;
            } else {
                seen_combinations.insert(key);
            }
        }
    }

    return is_duplicate;
}

vegaDataframe vegaDataframe::drop_duplicates(const std::vector<std::string>& subset, bool keep_first) const {
    auto duplicate_mask = duplicated(subset, keep_first);

    vegaDataframe result;
    result.data_features = data_features;
    result.column_types = column_types;

    for (size_t i = 0; i < data_values.size(); ++i) {
        if (!duplicate_mask[i]) {
            result.data_values.push_back(data_values[i]);
        }
    }

    result.update_stats_after_modification();
    return result;
}

// ============= RESHAPING OPERATIONS =============

vegaDataframe vegaDataframe::transpose() const {
    vegaDataframe result;

    // Transpose: rows become columns, columns become rows
    result.data_features.resize(data_values.size());
    for (size_t i = 0; i < data_values.size(); ++i) {
        result.data_features[i] = "row_" + std::to_string(i);
    }

    result.column_types.assign(result.data_features.size(), DataType::STRING);

    // Create transposed data
    for (size_t col = 0; col < data_features.size(); ++col) {
        std::vector<std::string> new_row;
        for (size_t row = 0; row < data_values.size(); ++row) {
            if (col < data_values[row].size()) {
                new_row.push_back(data_values[row][col]);
            } else {
                new_row.push_back("");
            }
        }
        result.data_values.push_back(new_row);
    }

    result.update_stats_after_modification();
    return result;
}

vegaDataframe vegaDataframe::copy() const {
    return *this;
}

bool vegaDataframe::empty() const {
    return data_values.empty() || data_features.empty();
}

bool vegaDataframe::equals(const vegaDataframe& other) const {
    return data_features == other.data_features &&
           data_values == other.data_values &&
           column_types == other.column_types;
}

std::vector<std::string> vegaDataframe::unique(const std::string& col_name) const {
    size_t col_idx = find_column_index(col_name);

    std::set<std::string> unique_set;
    for (const auto& row : data_values) {
        if (col_idx < row.size() && !row[col_idx].empty()) {
            unique_set.insert(row[col_idx]);
        }
    }

    return std::vector<std::string>(unique_set.begin(), unique_set.end());
}

// ============= EXPORT OPERATIONS =============

void vegaDataframe::to_csv(const std::string& filename, bool index, char sep) const {
    std::ofstream file(filename);
    if (!file) throw FILE_ERROR("Cannot create output file: " + filename);

    // Write header
    if (index) {
        file << "index" << sep;
    }

    for (size_t i = 0; i < data_features.size(); ++i) {
        file << data_features[i];
        if (i < data_features.size() - 1) file << sep;
    }
    file << "\n";

    // Write data
    for (size_t row_idx = 0; row_idx < data_values.size(); ++row_idx) {
        const auto& row = data_values[row_idx];

        if (index) {
            file << row_idx << sep;
        }

        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) file << sep;
        }
        file << "\n";
    }

    std::cout << "DataFrame exported to: " << filename << "\n";
}

void vegaDataframe::to_json(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file) throw FILE_ERROR("Cannot create JSON file: " + filename);

    file << "[\n";
    for (size_t row_idx = 0; row_idx < data_values.size(); ++row_idx) {
        const auto& row = data_values[row_idx];
        file << "  {\n";

        for (size_t col_idx = 0; col_idx < data_features.size(); ++col_idx) {
            file << "    \"" << data_features[col_idx] << "\": ";

            std::string value = (col_idx < row.size()) ? row[col_idx] : "";
            if (column_types[col_idx] == DataType::STRING) {
                file << "\"" << value << "\"";
            } else {
                file << (value.empty() ? "null" : value);
            }

            if (col_idx < data_features.size() - 1) file << ",";
            file << "\n";
        }

        file << "  }";
        if (row_idx < data_values.size() - 1) file << ",";
        file << "\n";
    }
    file << "]\n";

    std::cout << "DataFrame exported to JSON: " << filename << "\n";
}

void vegaDataframe::to_html(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file) throw FILE_ERROR("Cannot create HTML file: " + filename);

    file << "<!DOCTYPE html>\n<html>\n<head>\n";
    file << "<style>\ntable { border-collapse: collapse; width: 100%; }\n";
    file << "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n";
    file << "th { background-color: #f2f2f2; }\n</style>\n</head>\n<body>\n";
    file << "<table>\n<tr>\n";

    // Header
    for (const auto& col_name : data_features) {
        file << "<th>" << col_name << "</th>\n";
    }
    file << "</tr>\n";

    // Data rows
    for (const auto& row : data_values) {
        file << "<tr>\n";
        for (size_t i = 0; i < data_features.size(); ++i) {
            std::string value = (i < row.size()) ? row[i] : "";
            file << "<td>" << value << "</td>\n";
        }
        file << "</tr>\n";
    }

    file << "</table>\n</body>\n</html>\n";
    std::cout << "DataFrame exported to HTML: " << filename << "\n";
}

// ============= WINDOW FUNCTIONS =============

std::vector<double> vegaDataframe::rolling_mean(const std::string& col_name, size_t window) const {
    size_t col_idx = find_column_index(col_name);

    if (column_types[col_idx] == DataType::STRING)
        throw std::runtime_error("Cannot compute rolling mean for string column");

    std::vector<double> result;
    std::vector<double> values;

    // Extract numeric values
    for (const auto& row : data_values) {
        if (col_idx < row.size() && !row[col_idx].empty()) {
            try {
                values.push_back(std::stod(row[col_idx]));
            } catch (...) {
                values.push_back(std::numeric_limits<double>::quiet_NaN());
            }
        } else {
            values.push_back(std::numeric_limits<double>::quiet_NaN());
        }
    }

    // Calculate rolling mean
    for (size_t i = 0; i < values.size(); ++i) {
        if (i < window - 1) {
            result.push_back(std::numeric_limits<double>::quiet_NaN());
        } else {
            double sum = 0.0;
            size_t count = 0;
            for (size_t j = i - window + 1; j <= i; ++j) {
                if (!std::isnan(values[j])) {
                    sum += values[j];
                    count++;
                }
            }
            result.push_back(count > 0 ? sum / count : std::numeric_limits<double>::quiet_NaN());
        }
    }

    return result;
}

std::vector<double> vegaDataframe::rolling_sum(const std::string& col_name, size_t window) const {
    size_t col_idx = find_column_index(col_name);

    if (column_types[col_idx] == DataType::STRING)
        throw std::runtime_error("Cannot compute rolling sum for string column");

    std::vector<double> result;
    std::vector<double> values;

    for (const auto& row : data_values) {
        if (col_idx < row.size() && !row[col_idx].empty()) {
            try {
                values.push_back(std::stod(row[col_idx]));
            } catch (...) {
                values.push_back(0.0);
            }
        } else {
            values.push_back(0.0);
        }
    }

    for (size_t i = 0; i < values.size(); ++i) {
        if (i < window - 1) {
            result.push_back(std::numeric_limits<double>::quiet_NaN());
        } else {
            double sum = 0.0;
            for (size_t j = i - window + 1; j <= i; ++j) {
                sum += values[j];
            }
            result.push_back(sum);
        }
    }

    return result;
}

std::vector<double> vegaDataframe::rolling_std(const std::string& col_name, size_t window) const {
    size_t col_idx = find_column_index(col_name);

    if (column_types[col_idx] == DataType::STRING)
        throw std::runtime_error("Cannot compute rolling std for string column");

    std::vector<double> result;
    std::vector<double> values;

    for (const auto& row : data_values) {
        if (col_idx < row.size() && !row[col_idx].empty()) {
            try {
                values.push_back(std::stod(row[col_idx]));
            } catch (...) {
                values.push_back(std::numeric_limits<double>::quiet_NaN());
            }
        } else {
            values.push_back(std::numeric_limits<double>::quiet_NaN());
        }
    }

    for (size_t i = 0; i < values.size(); ++i) {
        if (i < window - 1) {
            result.push_back(std::numeric_limits<double>::quiet_NaN());
        } else {
            // Calculate mean for window
            double sum = 0.0;
            size_t count = 0;
            for (size_t j = i - window + 1; j <= i; ++j) {
                if (!std::isnan(values[j])) {
                    sum += values[j];
                    count++;
                }
            }

            if (count <= 1) {
                result.push_back(std::numeric_limits<double>::quiet_NaN());
                continue;
            }

            double mean = sum / count;

            // Calculate variance
            double variance_sum = 0.0;
            for (size_t j = i - window + 1; j <= i; ++j) {
                if (!std::isnan(values[j])) {
                    variance_sum += (values[j] - mean) * (values[j] - mean);
                }
            }

            double std_dev = std::sqrt(variance_sum / (count - 1));
            result.push_back(std_dev);
        }
    }

    return result;
}

std::vector<double> vegaDataframe::expanding_mean(const std::string& col_name) const {
    size_t col_idx = find_column_index(col_name);

    if (column_types[col_idx] == DataType::STRING)
        throw std::runtime_error("Cannot compute expanding mean for string column");

    std::vector<double> result;
    double running_sum = 0.0;
    size_t running_count = 0;

    for (const auto& row : data_values) {
        if (col_idx < row.size() && !row[col_idx].empty()) {
            try {
                double value = std::stod(row[col_idx]);
                running_sum += value;
                running_count++;
                result.push_back(running_sum / running_count);
            } catch (...) {
                result.push_back(running_count > 0 ? running_sum / running_count : std::numeric_limits<double>::quiet_NaN());
            }
        } else {
            result.push_back(running_count > 0 ? running_sum / running_count : std::numeric_limits<double>::quiet_NaN());
        }
    }

    return result;
}

std::vector<double> vegaDataframe::cumsum(const std::string& col_name) const {
    size_t col_idx = find_column_index(col_name);

    if (column_types[col_idx] == DataType::STRING)
        throw std::runtime_error("Cannot compute cumulative sum for string column");

    std::vector<double> result;
    double cumulative = 0.0;

    for (const auto& row : data_values) {
        if (col_idx < row.size() && !row[col_idx].empty()) {
            try {
                cumulative += std::stod(row[col_idx]);
            } catch (...) {}
        }
        result.push_back(cumulative);
    }

    return result;
}

std::vector<double> vegaDataframe::cumprod(const std::string& col_name) const {
    size_t col_idx = find_column_index(col_name);

    if (column_types[col_idx] == DataType::STRING)
        throw std::runtime_error("Cannot compute cumulative product for string column");

    std::vector<double> result;
    double cumulative = 1.0;

    for (const auto& row : data_values) {
        if (col_idx < row.size() && !row[col_idx].empty()) {
            try {
                cumulative *= std::stod(row[col_idx]);
            } catch (...) {}
        }
        result.push_back(cumulative);
    }

    return result;
}

std::vector<double> vegaDataframe::pct_change(const std::string& col_name, size_t periods) const {
    size_t col_idx = find_column_index(col_name);

    if (column_types[col_idx] == DataType::STRING)
        throw std::runtime_error("Cannot compute percent change for string column");

    std::vector<double> result;
    std::vector<double> values;

    // Extract values
    for (const auto& row : data_values) {
        if (col_idx < row.size() && !row[col_idx].empty()) {
            try {
                values.push_back(std::stod(row[col_idx]));
            } catch (...) {
                values.push_back(std::numeric_limits<double>::quiet_NaN());
            }
        } else {
            values.push_back(std::numeric_limits<double>::quiet_NaN());
        }
    }

    // Calculate percent change
    for (size_t i = 0; i < values.size(); ++i) {
        if (i < periods) {
            result.push_back(std::numeric_limits<double>::quiet_NaN());
        } else {
            double current = values[i];
            double previous = values[i - periods];

            if (std::isnan(current) || std::isnan(previous) || previous == 0.0) {
                result.push_back(std::numeric_limits<double>::quiet_NaN());
            } else {
                result.push_back((current - previous) / previous);
            }
        }
    }

    return result;
}

// ============= DATETIME OPERATIONS =============

vegaDataframe vegaDataframe::to_datetime(const std::string& col_name, const std::string& format) const {
    vegaDataframe result = *this;
    size_t col_idx = find_column_index(col_name);

    // Simple datetime conversion (basic implementation)
    for (auto& row : result.data_values) {
        if (col_idx < row.size() && !row[col_idx].empty()) {
            // This is a simplified implementation
            // In a full implementation, you'd use a proper datetime parsing library
            std::string& date_str = row[col_idx];
            // For now, just validate it looks like a date
            if (date_str.find('-') != std::string::npos || date_str.find('/') != std::string::npos) {
                // Keep as is - already looks like a date
            }
        }
    }

    result.column_types[col_idx] = DataType::STRING; // Would be DateTime in full implementation
    return result;
}

std::vector<int> vegaDataframe::dt_year(const std::string& col_name) const {
    size_t col_idx = find_column_index(col_name);
    std::vector<int> years;

    for (const auto& row : data_values) {
        if (col_idx < row.size() && !row[col_idx].empty()) {
            std::string date_str = row[col_idx];
            // Simple year extraction (assumes YYYY-MM-DD or YYYY/MM/DD format)
            if (date_str.length() >= 4) {
                try {
                    int year = std::stoi(date_str.substr(0, 4));
                    years.push_back(year);
                } catch (...) {
                    years.push_back(0);
                }
            } else {
                years.push_back(0);
            }
        } else {
            years.push_back(0);
        }
    }

    return years;
}

std::vector<int> vegaDataframe::dt_month(const std::string& col_name) const {
    size_t col_idx = find_column_index(col_name);
    std::vector<int> months;

    for (const auto& row : data_values) {
        if (col_idx < row.size() && !row[col_idx].empty()) {
            std::string date_str = row[col_idx];
            // Extract month (assumes YYYY-MM-DD or YYYY/MM/DD format)
            size_t first_sep = date_str.find_first_of("-/");
            if (first_sep != std::string::npos && first_sep + 3 <= date_str.length()) {
                try {
                    int month = std::stoi(date_str.substr(first_sep + 1, 2));
                    months.push_back(month);
                } catch (...) {
                    months.push_back(0);
                }
            } else {
                months.push_back(0);
            }
        } else {
            months.push_back(0);
        }
    }

    return months;
}

std::vector<int> vegaDataframe::dt_day(const std::string& col_name) const {
    size_t col_idx = find_column_index(col_name);
    std::vector<int> days;

    for (const auto& row : data_values) {
        if (col_idx < row.size() && !row[col_idx].empty()) {
            std::string date_str = row[col_idx];
            // Extract day (assumes YYYY-MM-DD or YYYY/MM/DD format)
            size_t first_sep = date_str.find_first_of("-/");
            if (first_sep != std::string::npos) {
                size_t second_sep = date_str.find_first_of("-/", first_sep + 1);
                if (second_sep != std::string::npos && second_sep + 3 <= date_str.length()) {
                    try {
                        int day = std::stoi(date_str.substr(second_sep + 1, 2));
                        days.push_back(day);
                    } catch (...) {
                        days.push_back(0);
                    }
                } else {
                    days.push_back(0);
                }
            } else {
                days.push_back(0);
            }
        } else {
            days.push_back(0);
        }
    }

    return days;
}

std::vector<int> vegaDataframe::dt_dayofweek(const std::string& col_name) const {
    // Simplified implementation - would need proper date parsing in production
    std::vector<int> dayofweek(data_values.size(), 0);

    // This would require a proper datetime library to implement correctly
    // For now, return placeholder values
    for (size_t i = 0; i < data_values.size(); ++i) {
        dayofweek[i] = i % 7; // Placeholder: cycle through 0-6
    }

    return dayofweek;
}

// ============= ARITHMETIC OPERATIONS =============

vegaDataframe vegaDataframe::add(const vegaDataframe& other) const {
    if (shape() != other.shape()) {
        throw std::runtime_error("DataFrames must have same shape for arithmetic operations");
    }

    vegaDataframe result = *this;

    for (size_t i = 0; i < data_values.size(); ++i) {
        for (size_t j = 0; j < data_values[i].size(); ++j) {
            if (j < other.data_values[i].size() &&
                column_types[j] != DataType::STRING &&
                other.column_types[j] != DataType::STRING) {
                try {
                    double val1 = safe_stod(data_values[i][j]);
                    double val2 = safe_stod(other.data_values[i][j]);
                    result.data_values[i][j] = std::to_string(val1 + val2);
                } catch (...) {
                    result.data_values[i][j] = "";
                }
            }
        }
    }

    return result;
}

vegaDataframe vegaDataframe::subtract(const vegaDataframe& other) const {
    if (shape() != other.shape()) {
        throw std::runtime_error("DataFrames must have same shape for arithmetic operations");
    }

    vegaDataframe result = *this;

    for (size_t i = 0; i < data_values.size(); ++i) {
        for (size_t j = 0; j < data_values[i].size(); ++j) {
            if (j < other.data_values[i].size() &&
                column_types[j] != DataType::STRING &&
                other.column_types[j] != DataType::STRING) {
                try {
                    double val1 = safe_stod(data_values[i][j]);
                    double val2 = safe_stod(other.data_values[i][j]);
                    result.data_values[i][j] = std::to_string(val1 - val2);
                } catch (...) {
                    result.data_values[i][j] = "";
                }
            }
        }
    }

    return result;
}

vegaDataframe vegaDataframe::multiply(const vegaDataframe& other) const {
    if (shape() != other.shape()) {
        throw std::runtime_error("DataFrames must have same shape for arithmetic operations");
    }

    vegaDataframe result = *this;

    for (size_t i = 0; i < data_values.size(); ++i) {
        for (size_t j = 0; j < data_values[i].size(); ++j) {
            if (j < other.data_values[i].size() &&
                column_types[j] != DataType::STRING &&
                other.column_types[j] != DataType::STRING) {
                try {
                    double val1 = safe_stod(data_values[i][j]);
                    double val2 = safe_stod(other.data_values[i][j]);
                    result.data_values[i][j] = std::to_string(val1 * val2);
                } catch (...) {
                    result.data_values[i][j] = "";
                }
            }
        }
    }

    return result;
}

vegaDataframe vegaDataframe::divide(const vegaDataframe& other) const {
    if (shape() != other.shape()) {
        throw std::runtime_error("DataFrames must have same shape for arithmetic operations");
    }

    vegaDataframe result = *this;

    for (size_t i = 0; i < data_values.size(); ++i) {
        for (size_t j = 0; j < data_values[i].size(); ++j) {
            if (j < other.data_values[i].size() &&
                column_types[j] != DataType::STRING &&
                other.column_types[j] != DataType::STRING) {
                try {
                    double val1 = safe_stod(data_values[i][j]);
                    double val2 = safe_stod(other.data_values[i][j]);
                    if (val2 != 0.0) {
                        result.data_values[i][j] = std::to_string(val1 / val2);
                    } else {
                        result.data_values[i][j] = "inf";
                    }
                } catch (...) {
                    result.data_values[i][j] = "";
                }
            }
        }
    }

    return result;
}

vegaDataframe vegaDataframe::add_scalar(double value) const {
    vegaDataframe result = *this;

    for (size_t i = 0; i < data_values.size(); ++i) {
        for (size_t j = 0; j < data_values[i].size(); ++j) {
            if (column_types[j] != DataType::STRING) {
                try {
                    double val = safe_stod(data_values[i][j]);
                    result.data_values[i][j] = std::to_string(val + value);
                } catch (...) {
                    result.data_values[i][j] = data_values[i][j];
                }
            }
        }
    }

    return result;
}

vegaDataframe vegaDataframe::multiply_scalar(double value) const {
    vegaDataframe result = *this;

    for (size_t i = 0; i < data_values.size(); ++i) {
        for (size_t j = 0; j < data_values[i].size(); ++j) {
            if (column_types[j] != DataType::STRING) {
                try {
                    double val = safe_stod(data_values[i][j]);
                    result.data_values[i][j] = std::to_string(val * value);
                } catch (...) {
                    result.data_values[i][j] = data_values[i][j];
                }
            }
        }
    }

    return result;
}

// ============= COMPARISON OPERATIONS =============

std::vector<std::vector<bool>> vegaDataframe::eq(const vegaDataframe& other) const {
    if (shape() != other.shape()) {
        throw std::runtime_error("DataFrames must have same shape for comparison");
    }

    std::vector<std::vector<bool>> result(data_values.size());

    for (size_t i = 0; i < data_values.size(); ++i) {
        result[i].resize(data_values[i].size());
        for (size_t j = 0; j < data_values[i].size(); ++j) {
            if (j < other.data_values[i].size()) {
                result[i][j] = (data_values[i][j] == other.data_values[i][j]);
            } else {
                result[i][j] = false;
            }
        }
    }

    return result;
}

std::vector<std::vector<bool>> vegaDataframe::ne(const vegaDataframe& other) const {
    auto eq_result = eq(other);

    for (auto& row : eq_result) {
        for (auto val : row) {
            val = !val;
        }
    }

    return eq_result;
}

std::vector<std::vector<bool>> vegaDataframe::lt(const vegaDataframe& other) const {
    if (shape() != other.shape()) {
        throw std::runtime_error("DataFrames must have same shape for comparison");
    }

    std::vector<std::vector<bool>> result(data_values.size());

    for (size_t i = 0; i < data_values.size(); ++i) {
        result[i].resize(data_values[i].size());
        for (size_t j = 0; j < data_values[i].size(); ++j) {
            if (j < other.data_values[i].size() &&
                column_types[j] != DataType::STRING &&
                other.column_types[j] != DataType::STRING) {
                try {
                    double val1 = safe_stod(data_values[i][j]);
                    double val2 = safe_stod(other.data_values[i][j]);
                    result[i][j] = (val1 < val2);
                } catch (...) {
                    result[i][j] = false;
                }
            } else {
                result[i][j] = (data_values[i][j] < other.data_values[i][j]);
            }
        }
    }

    return result;
}

std::vector<std::vector<bool>> vegaDataframe::le(const vegaDataframe& other) const {
    auto lt_result = lt(other);
    auto eq_result = eq(other);

    for (size_t i = 0; i < lt_result.size(); ++i) {
        for (size_t j = 0; j < lt_result[i].size(); ++j) {
            lt_result[i][j] = lt_result[i][j] || eq_result[i][j];
        }
    }

    return lt_result;
}

std::vector<std::vector<bool>> vegaDataframe::gt(const vegaDataframe& other) const {
    auto le_result = le(other);

    for (auto& row : le_result) {
        for (auto val : row) {
            val = !val;
        }
    }

    return le_result;
}

std::vector<std::vector<bool>> vegaDataframe::ge(const vegaDataframe& other) const {
    auto lt_result = lt(other);

    for (auto& row : lt_result) {
        for (auto val : row) {
            val = !val;
        }
    }

    return lt_result;
}

// ============= ADDITIONAL UTILITY OPERATIONS =============

vegaDataframe vegaDataframe::where(const std::function<bool(const std::vector<std::string>&)>& condition, const std::string& other) const {
    vegaDataframe result = *this;

    for (size_t i = 0; i < data_values.size(); ++i) {
        if (!condition(data_values[i])) {
            // Replace entire row with 'other' value
            for (auto& cell : result.data_values[i]) {
                cell = other;
            }
        }
    }

    return result;
}

vegaDataframe vegaDataframe::astype(const std::string& col_name, DataType dtype) {
    size_t col_idx = find_column_index(col_name);
    column_types[col_idx] = dtype;

    // Could add conversion logic here if needed
    return *this;
}

vegaDataframe vegaDataframe::reset_index(bool drop) const {
    vegaDataframe result = *this;

    if (!drop) {
        // Add index column
        std::vector<std::string> index_col;
        for (size_t i = 0; i < data_values.size(); ++i) {
            index_col.push_back(std::to_string(i));
        }
        result.insert_column(0, "index", index_col);
    }

    return result;
}

vegaDataframe vegaDataframe::set_index(const std::string& col_name) const {
    // In a full implementation, this would set the specified column as the index
    // For now, just return a copy
    vegaDataframe result = *this;
    // Implementation would involve index management
    return result;
}

void vegaDataframe::to_excel(const std::string& filename) const {
    // This would require an Excel library like libxl or similar
    // For now, export as CSV with .xlsx extension
    std::string csv_filename = filename;
    if (csv_filename.substr(csv_filename.length() - 5) == ".xlsx") {
        csv_filename = csv_filename.substr(0, csv_filename.length() - 5) + ".csv";
    }

    to_csv(csv_filename);
    std::cout << "Note: Exported as CSV instead of Excel format. Filename: " << csv_filename << "\n";
}

// ============= PIVOT OPERATIONS (BASIC IMPLEMENTATION) =============

vegaDataframe vegaDataframe::pivot_table(const std::string& values, const std::string& index, const std::string& columns) const {
    // Simplified pivot table implementation
    size_t values_idx = find_column_index(values);
    size_t index_idx = find_column_index(index);
    size_t columns_idx = find_column_index(columns);

    vegaDataframe result;

    // Get unique values for index and columns
    auto index_unique = unique(index);
    auto columns_unique = unique(columns);

    // Set up result structure
    result.data_features.push_back(index);
    for (const auto& col_val : columns_unique) {
        result.data_features.push_back(col_val);
    }

    result.column_types.assign(result.data_features.size(), DataType::FLOAT);
    result.column_types[0] = DataType::STRING; // Index column

    // Create pivot data
    for (const auto& idx_val : index_unique) {
        std::vector<std::string> result_row;
        result_row.push_back(idx_val);

        for (const auto& col_val : columns_unique) {
            double sum = 0.0;
            size_t count = 0;

            // Find matching rows and aggregate
            for (const auto& row : data_values) {
                if (index_idx < row.size() && columns_idx < row.size() && values_idx < row.size() &&
                    row[index_idx] == idx_val && row[columns_idx] == col_val) {
                    try {
                        sum += std::stod(row[values_idx]);
                        count++;
                    } catch (...) {}
                }
            }

            if (count > 0) {
                result_row.push_back(std::to_string(sum / count)); // Mean aggregation
            } else {
                result_row.push_back("");
            }
        }

        result.data_values.push_back(result_row);
    }

    result.update_stats_after_modification();
    return result;
}

vegaDataframe vegaDataframe::pivot(const std::string& index, const std::string& columns, const std::string& values) const {
    return pivot_table(values, index, columns);
}

vegaDataframe vegaDataframe::melt(const std::vector<std::string>& id_vars, const std::vector<std::string>& value_vars) const {
    vegaDataframe result;

    // Set up result columns
    for (const auto& id_var : id_vars) {
        result.data_features.push_back(id_var);
        result.column_types.push_back(column_types[find_column_index(id_var)]);
    }
    result.data_features.push_back("variable");
    result.data_features.push_back("value");
    result.column_types.push_back(DataType::STRING);
    result.column_types.push_back(DataType::STRING);

    // Determine which columns to melt
    std::vector<std::string> cols_to_melt = value_vars;
    if (cols_to_melt.empty()) {
        // Melt all columns except id_vars
        for (const auto& feature : data_features) {
            if (std::ranges::find(id_vars, feature) == id_vars.end()) {
                cols_to_melt.push_back(feature);
            }
        }
    }

    // Create melted data
    for (const auto& row : data_values) {
        for (const auto& col_to_melt : cols_to_melt) {
            std::vector<std::string> result_row;

            // Add id_var values
            for (const auto& id_var : id_vars) {
                size_t id_idx = find_column_index(id_var);
                if (id_idx < row.size()) {
                    result_row.push_back(row[id_idx]);
                } else {
                    result_row.push_back("");
                }
            }

            // Add variable name and value
            result_row.push_back(col_to_melt);
            size_t val_idx = find_column_index(col_to_melt);
            if (val_idx < row.size()) {
                result_row.push_back(row[val_idx]);
            } else {
                result_row.push_back("");
            }

            result.data_values.push_back(result_row);
        }
    }

    result.update_stats_after_modification();
    return result;
}

vegaDataframe vegaDataframe::stack() const {
    // Basic stack implementation - convert columns to a single column with multi-level index
    vegaDataframe result;
    result.data_features = {"level_0", "level_1", "value"};
    result.column_types = {DataType::INT, DataType::STRING, DataType::STRING};

    for (size_t row_idx = 0; row_idx < data_values.size(); ++row_idx) {
        for (size_t col_idx = 0; col_idx < data_features.size(); ++col_idx) {
            std::vector<std::string> stacked_row;
            stacked_row.push_back(std::to_string(row_idx));
            stacked_row.push_back(data_features[col_idx]);

            if (col_idx < data_values[row_idx].size()) {
                stacked_row.push_back(data_values[row_idx][col_idx]);
            } else {
                stacked_row.push_back("");
            }

            result.data_values.push_back(stacked_row);
        }
    }

    result.update_stats_after_modification();
    return result;
}

vegaDataframe vegaDataframe::unstack() const {
    // Basic unstack implementation - reverse of stack
    // This is a simplified version
    return transpose();
}

vegaDataframe vegaDataframe::reindex(const std::vector<size_t>& new_index) const {
    vegaDataframe result;
    result.data_features = data_features;
    result.column_types = column_types;

    for (size_t idx : new_index) {
        if (idx < data_values.size()) {
            result.data_values.push_back(data_values[idx]);
        } else {
            // Add empty row for out-of-bounds indices
            std::vector<std::string> empty_row(data_features.size(), "");
            result.data_values.push_back(empty_row);
        }
    }

    result.update_stats_after_modification();
    return result;
}