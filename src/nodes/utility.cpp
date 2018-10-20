/**
 * @file	utility.cpp
 * @author	Carroll Vance
 * @brief	Node Utility Functions
 *
 * Copyright (c) 2018 Carroll Vance.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <fstream>
#include <streambuf>

#include "utility.h"

std::vector<std::string> load_class_descriptions(std::string filename) {

  std::vector<std::string> classes;

  std::ifstream words_file(filename.c_str());
  std::string text((std::istreambuf_iterator<char>(words_file)),
                   std::istreambuf_iterator<char>());

  std::string delim = "\n";
  auto start = 0U;
  auto end = text.find(delim);
  int index = 0;
  while (end != std::string::npos) {

    classes.push_back(text.substr(start, end - start));

    start = end + delim.length();
    end = text.find(delim, start);
    index++;
  }

  return classes;
}
