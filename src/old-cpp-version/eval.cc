#include <stdlib.h>
#include <iomanip>

#include <sstream>

#include <eval.h>

double calc_ratio(unsigned int numerator, unsigned int denominator) {
  if (denominator > 0)
    return static_cast<double>(numerator) / static_cast<double>(denominator);
  else
    return 0.0;
}

double Evaluator::p(bool start) {
  if (start) return calc_ratio(starts_matched, starts_total_guess);
  else return calc_ratio(matched, total_guess);
}
double Evaluator::r(bool start) {
  if (start) return calc_ratio(starts_matched, starts_total_gold);
  else return calc_ratio(matched, total_gold);
}
double Evaluator::f(bool start) {
  unsigned int to_check = matched;
  if (start) to_check = starts_matched;
  if (to_check > 0) {
    double precision = p(start);
    double recall = r(start);
    return 2 * precision * recall / (precision + recall);
  } else {
    return 0.0;
  }
}

string Evaluator::to_string() {
  ostringstream output;
  output << "Eval:"
    << " match:" << matched
    << " gold:" << total_gold
    << " guess:" << total_guess
    << setprecision(3)
    << " starts-p,r,f: " << p(true) << " " << r(true) << " " << f(true)
    << " p,r,f: " << p() << " " << r() << " " << f();

  return output.str();
}
