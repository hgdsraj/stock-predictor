import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { Eye, AlertTriangle } from "lucide-react";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  flexRender,
  type SortingState,
  type ColumnDef,
} from "@tanstack/react-table";
import { ArrowUpDown, Search as SearchIcon } from "lucide-react";
import { api } from "@/api/client";
import { Card, CardContent, CardHeader, CardTitle, CardSubtitle } from "@/components/ui/Card";
import { Input } from "@/components/ui/Input";
import { Table, THead, TBody, TR, TH, TD } from "@/components/ui/Table";
import {
  formatCompactNumber,
  formatDate,
  formatNumber,
} from "@/lib/format";
import { cn } from "@/lib/cn";

interface Row {
  ticker: string;
  sector: string | null;
  industry: string | null;
  last_price: number | null;
  market_cap: number | null;
  last_updated: string | null;
}

export function Screener() {
  const { data, isLoading } = useQuery({ queryKey: ["tickers"], queryFn: api.tickers });
  const watchlist = useQuery({ queryKey: ["watchlist"], queryFn: api.watchlist });
  const [sorting, setSorting] = useState<SortingState>([{ id: "market_cap", desc: true }]);
  const [search, setSearch] = useState("");
  const [sectorFilter, setSectorFilter] = useState<string>("All");

  const sectors = useMemo(() => {
    const s = new Set<string>();
    (data ?? []).forEach((d) => d.sector && s.add(d.sector));
    return ["All", ...Array.from(s).sort()];
  }, [data]);

  const filteredData = useMemo(() => {
    const rows = data ?? [];
    return rows.filter((r) => {
      if (sectorFilter !== "All" && r.sector !== sectorFilter) return false;
      if (search) {
        const q = search.toLowerCase();
        if (!r.ticker.toLowerCase().includes(q) && !(r.industry?.toLowerCase().includes(q))) return false;
      }
      return true;
    });
  }, [data, sectorFilter, search]);

  const columns = useMemo<ColumnDef<Row>[]>(
    () => [
      {
        accessorKey: "ticker",
        header: "Ticker",
        cell: (info) => (
          <Link
            to={`/ticker/${encodeURIComponent(info.getValue<string>())}`}
            className="font-semibold hover:underline"
          >
            {info.getValue<string>()}
          </Link>
        ),
      },
      { accessorKey: "sector", header: "Sector", cell: (info) => info.getValue() ?? "—" },
      { accessorKey: "industry", header: "Industry", cell: (info) => info.getValue() ?? "—" },
      {
        accessorKey: "last_price",
        header: "Last",
        cell: (info) => formatNumber(info.getValue<number | null>(), { style: "currency", currency: "USD" }),
      },
      {
        accessorKey: "market_cap",
        header: "Mkt Cap",
        cell: (info) => formatCompactNumber(info.getValue<number | null>()),
      },
      {
        accessorKey: "last_updated",
        header: "As of",
        cell: (info) => formatDate(info.getValue<string | null>()),
      },
    ],
    [],
  );

  const table = useReactTable({
    data: filteredData,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
  });

  const watchlistItems = watchlist.data ?? [];

  return (
    <div className="space-y-4">
      {watchlistItems.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Eye className="h-4 w-4 text-muted-foreground" />
              <CardTitle>Watchlist</CardTitle>
            </div>
            <CardSubtitle>
              External instruments tracked for context. Not included in the model's universe.
            </CardSubtitle>
          </CardHeader>
          <CardContent className="p-0">
            <Table>
              <THead>
                <TR>
                  <TH>Ticker</TH>
                  <TH>Label</TH>
                  <TH>Category</TH>
                  <TH>Last</TH>
                  <TH>As of</TH>
                  <TH>Note</TH>
                </TR>
              </THead>
              <TBody>
                {watchlistItems.map((w) => (
                  <TR key={w.ticker}>
                    <TD>
                      <Link
                        to={`/ticker/${encodeURIComponent(w.ticker)}`}
                        className="font-semibold hover:underline"
                      >
                        {w.ticker}
                      </Link>
                    </TD>
                    <TD>{w.label ?? "—"}</TD>
                    <TD>
                      <span className="badge">{w.category ?? "—"}</span>
                    </TD>
                    <TD>{formatNumber(w.last_price, { style: "currency", currency: "USD" })}</TD>
                    <TD>{formatDate(w.last_updated)}</TD>
                    <TD className="text-xs text-muted-foreground">
                      {w.note?.startsWith("WARNING") ? (
                        <span className="inline-flex items-center gap-1 text-amber-600">
                          <AlertTriangle className="h-3 w-3" />
                          {w.note}
                        </span>
                      ) : (
                        w.note ?? "—"
                      )}
                    </TD>
                  </TR>
                ))}
              </TBody>
            </Table>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <CardTitle>Universe screener</CardTitle>
              <CardSubtitle>
                {data ? `${filteredData.length} of ${data.length} tickers` : "Loading…"}
              </CardSubtitle>
            </div>
            <div className="flex flex-wrap gap-2">
              <div className="relative">
                <SearchIcon className="pointer-events-none absolute left-2 top-2 h-4 w-4 text-muted-foreground" />
                <Input
                  className="pl-8"
                  placeholder="Search ticker or industry"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                />
              </div>
              <select
                value={sectorFilter}
                onChange={(e) => setSectorFilter(e.target.value)}
                className="h-9 rounded-md border border-border bg-background px-3 text-sm"
              >
                {sectors.map((s) => (
                  <option key={s}>{s}</option>
                ))}
              </select>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          {isLoading ? (
            <div className="p-6 text-sm text-muted-foreground">Loading…</div>
          ) : filteredData.length === 0 ? (
            <div className="p-6 text-sm text-muted-foreground">No tickers match your filters.</div>
          ) : (
            <Table>
              <THead>
                {table.getHeaderGroups().map((hg) => (
                  <TR key={hg.id}>
                    {hg.headers.map((header) => (
                      <TH
                        key={header.id}
                        onClick={header.column.getToggleSortingHandler()}
                        className={cn(
                          "cursor-pointer select-none whitespace-nowrap",
                          header.column.getIsSorted() && "text-foreground",
                        )}
                      >
                        <span className="inline-flex items-center gap-1">
                          {flexRender(header.column.columnDef.header, header.getContext())}
                          <ArrowUpDown className="h-3 w-3 opacity-60" />
                        </span>
                      </TH>
                    ))}
                  </TR>
                ))}
              </THead>
              <TBody>
                {table.getRowModel().rows.slice(0, 200).map((row) => (
                  <TR key={row.id}>
                    {row.getVisibleCells().map((cell) => (
                      <TD key={cell.id}>
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </TD>
                    ))}
                  </TR>
                ))}
              </TBody>
            </Table>
          )}
        </CardContent>
      </Card>
      {filteredData.length > 200 && (
        <p className="text-center text-xs text-muted-foreground">
          Showing first 200 rows. Refine with search / sector to see more.
        </p>
      )}
    </div>
  );
}
