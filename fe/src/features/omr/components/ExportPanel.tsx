import { jsPDF } from "jspdf";
import * as XLSX from "xlsx";
import type { ExportRecordRow, GradeRecord, TestCardItem } from "../types";
import {
  buildExportRecordRows,
  downloadBlobFile,
  toSafeFileName,
} from "../utils";

type ExportPanelProps = {
  selectedTest: TestCardItem | null;
  gradeRecords: GradeRecord[];
  onError: (message: string) => void;
  onSuccess: (message: string) => void;
};

export default function ExportPanel({
  selectedTest,
  gradeRecords,
  onError,
  onSuccess,
}: ExportPanelProps) {
  function getExportRows(fileType: "Excel" | "PDF") {
    if (!selectedTest) {
      onError("Chưa chọn bài kiểm tra để xuất file.");
      return null;
    }

    const exportRows = buildExportRecordRows(gradeRecords);
    if (exportRows.length === 0) {
      onError(`Chưa có dữ liệu để xuất ${fileType}.`);
      return null;
    }

    return exportRows;
  }

  function exportExcel() {
    const exportRows = getExportRows("Excel");
    if (!selectedTest || !exportRows) return;

    const sheetData: string[][] = [
      ["MSSV", "Mã đề", "Điểm"],
      ...exportRows.map((row) => [row.mssv, row.examCode, row.score]),
    ];

    const workbook = XLSX.utils.book_new();
    const worksheet = XLSX.utils.aoa_to_sheet(sheetData);
    worksheet["!cols"] = [{ wch: 20 }, { wch: 12 }, { wch: 10 }];
    XLSX.utils.book_append_sheet(workbook, worksheet, "Danh_sach_cham");

    const fileBuffer = XLSX.write(workbook, { bookType: "xlsx", type: "array" });
    const blob = new Blob(
      [fileBuffer],
      { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" }
    );

    downloadBlobFile(`${toSafeFileName(selectedTest.title)}_danh_sach_cham.xlsx`, blob);
    onSuccess(`Đã xuất Excel ${exportRows.length} bản ghi.`);
  }

  function exportPdfSummary() {
    const exportRows = getExportRows("PDF");
    if (!selectedTest || !exportRows) return;

    const doc = new jsPDF({ unit: "pt", format: "a4" });
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const marginX = 34;
    const tableBottomPadding = 36;
    const rowHeight = 24;
    const colWidths = [pageWidth - (marginX * 2) - 188, 98, 90];
    const headers = ["MSSV", "Mã đề", "Điểm"];
    let cursorY = 42;

    const drawTableHeader = () => {
      let x = marginX;
      doc.setFillColor(235, 243, 252);
      doc.setDrawColor(171, 195, 220);
      doc.setTextColor(20, 50, 79);
      doc.setFont("helvetica", "bold");
      headers.forEach((label, idx) => {
        const width = colWidths[idx];
        doc.rect(x, cursorY, width, rowHeight, "FD");
        doc.text(label, x + 6, cursorY + 16);
        x += width;
      });
      cursorY += rowHeight;
      doc.setTextColor(29, 49, 68);
      doc.setFont("helvetica", "normal");
    };

    const drawTableRow = (row: ExportRecordRow) => {
      const values = [row.mssv, row.examCode, row.score];
      let x = marginX;

      values.forEach((value, idx) => {
        const width = colWidths[idx];
        doc.setDrawColor(196, 214, 232);
        doc.rect(x, cursorY, width, rowHeight);
        const fragments = doc.splitTextToSize(String(value), width - 10);
        const firstLine = Array.isArray(fragments) ? (fragments[0] || "") : String(fragments);
        doc.text(firstLine, x + 6, cursorY + 16);
        x += width;
      });

      cursorY += rowHeight;
    };

    doc.setFont("helvetica", "bold");
    doc.setFontSize(13);
    doc.setTextColor(17, 43, 68);
    doc.text(`Báo cáo OMR - ${selectedTest.title}`, marginX, cursorY);

    cursorY += 18;
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    doc.text(`Ngày xuất: ${new Date().toLocaleString("vi-VN")}`, marginX, cursorY);
    doc.text(`Số bản ghi: ${exportRows.length}`, marginX + 250, cursorY);

    cursorY += 18;
    drawTableHeader();

    exportRows.forEach((row) => {
      if (cursorY + rowHeight > pageHeight - tableBottomPadding) {
        doc.addPage();
        cursorY = 42;
        drawTableHeader();
      }
      drawTableRow(row);
    });

    doc.save(`${toSafeFileName(selectedTest.title)}_danh_sach_cham.pdf`);
    onSuccess(`Đã xuất PDF ${exportRows.length} bản ghi.`);
  }

  return (
    <div className="export-wrap">
      <button className="export-btn tap-feedback" onClick={exportPdfSummary}>📄 Xuất PDF</button>
      <button className="export-btn tap-feedback" onClick={exportExcel}>📊 Xuất Excel</button>
    </div>
  );
}
