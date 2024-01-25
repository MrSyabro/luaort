---@meta

---@class PNGUtils
local M = {}

---Читает и декодирует png
---@param inputfile any
---@return number width
---@return number height
---@return string data
function M.read(inputfile)
end

---Кодирует и записывает png
---@param data string данные для записи
---@param height integer
---@param width integer
---@param outputfile string имя файла
function M.write(data, height, width, outputfile)
end

---convert input from CHW format to HWC format
---@param data string CHW data
---@return string data HWC data
function M.chw2hwc(data)
end

---convert input from HWC format to CHW format
---@param data string
---@return string data
function M.hwc2chw(data)
end

return M